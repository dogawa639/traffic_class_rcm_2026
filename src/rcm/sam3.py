"""SAM3 text-guided segmentation features for road links.

Uses Ultralytics SAM3 (https://docs.ultralytics.com/models/sam-3) to segment
objects in aerial photographs by text prompt (e.g. "road", "tree", "sidewalk").
Computes per-link coverage ratios: the fraction of pixels within the link's
buffer zone that belong to the segmented class.

Results are cached on disk to avoid redundant GPU inference:
- ``{cache_dir}/{word}/mask.npy``          — full (H, W) bool mask per word
- ``{cache_dir}/{word}/area_ratios.parquet`` — per-link area ratios

Re-running only recomputes what is not already cached.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import LineString

from rcm.entities import Link, Node

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _lon_lat_to_pixel(
    lon: float,
    lat: float,
    transform: list[float],
) -> tuple[float, float]:
    """Convert geographic coordinates to pixel (col, row).

    The affine transform follows GDAL/rasterio convention stored in the
    geo.json sidecar:  ``[px_w, 0, lon0, 0, px_h, lat0]``
    where ``px_h`` is negative (north-to-south).
    """
    px_w = transform[0]
    lon0 = transform[2]
    px_h = transform[4]
    lat0 = transform[5]
    col = (lon - lon0) / px_w
    row = (lat - lat0) / px_h
    return col, row


def _compute_link_overlaps_cropped(
    lk: Link,
    node_lookup: dict[int, Node],
    transform: list[float],
    img_h: int,
    img_w: int,
    buffer_px: float,
    seg_masks: list[np.ndarray],
) -> list[float]:
    """Compute overlap ratios between a link buffer and each segmentation mask.

    Only the bounding-box crop of the buffer polygon is rasterised, so memory
    usage is proportional to link length rather than the full image size.

    Parameters
    ----------
    seg_masks:
        List of (H, W) bool arrays — one per keyword.

    Returns
    -------
    list[float]
        Coverage ratio in [0, 1] for each entry in *seg_masks*.
        Returns all-zeros if the link falls outside the image or has no area.
    """
    zeros = [0.0] * len(seg_masks)

    from_node = node_lookup.get(lk.from_node)
    to_node = node_lookup.get(lk.to_node)
    if from_node is None or to_node is None:
        return zeros

    col1, row1 = _lon_lat_to_pixel(from_node.lon, from_node.lat, transform)
    col2, row2 = _lon_lat_to_pixel(to_node.lon, to_node.lat, transform)

    # Degenerate segment: use a circular buffer around midpoint
    if abs(col1 - col2) < 0.5 and abs(row1 - row2) < 0.5:
        line = LineString([(col1, row1), (col1 + 0.5, row1)])
    else:
        line = LineString([(col1, row1), (col2, row2)])

    buffered = line.buffer(buffer_px)
    minx, miny, maxx, maxy = buffered.bounds

    # Clip bounding box to image bounds
    x0 = max(0, int(minx))
    y0 = max(0, int(miny))
    x1 = min(img_w, int(maxx) + 1)
    y1 = min(img_h, int(maxy) + 1)

    if x0 >= x1 or y0 >= y1:
        return zeros

    # Rasterise only the bounding-box crop (a few KB, not the full image)
    crop_w = x1 - x0
    crop_h = y1 - y0
    img_mask = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(img_mask)
    exterior = [(c[0] - x0, c[1] - y0) for c in buffered.exterior.coords]
    draw.polygon(exterior, fill=1)
    buf_crop = np.array(img_mask, dtype=bool)

    area = float(buf_crop.sum())
    if area == 0:
        return zeros

    ratios = []
    for seg_mask in seg_masks:
        seg_crop = seg_mask[y0:y1, x0:x1]
        overlap = float((seg_crop & buf_crop).sum())
        ratios.append(overlap / area)

    return ratios


# ---------------------------------------------------------------------------
# Segmentation mask
# ---------------------------------------------------------------------------


def _run_sam3(
    aerial_photo_path: Path,
    word: str,
    model_name: str,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Run SAM3 text-prompted segmentation and return a (H, W) bool mask.

    Combines masks from all detected instances via logical OR.
    """
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for SAM3 features. "
            "Install with: pip install 'rcm-exercise[sam3]'"
        ) from exc

    log.info("Running SAM3 (%s) for word=%r …", model_name, word)

    overrides = dict(conf=0.25, task="segment", mode="predict", model=model_name, half=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(str(aerial_photo_path))
    results = predictor(text=[word])

    combined = np.zeros((img_h, img_w), dtype=bool)
    for result in results:
        if result.masks is None:
            continue
        # masks.data: torch.Tensor (N, H, W)
        masks_np = result.masks.data.cpu().numpy().astype(bool)
        for m in masks_np:
            # Resize to full image if necessary
            if m.shape != (img_h, img_w):
                from PIL import Image as _Image
                m_img = _Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (img_w, img_h), resample=_Image.NEAREST
                )
                m = np.array(m_img) > 127
            combined |= m

    if not combined.any():
        warnings.warn(
            f"SAM3 found no segments for word={word!r}. "
            "The word may not match any visible objects in the aerial photo.",
            stacklevel=3,
        )
    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_sam3_features(
    aerial_photo_path: str | Path,
    geo_json_path: str | Path,
    words: list[str],
    links: list[Link],
    node_lookup: dict[int, Node],
    cache_dir: str | Path,
    *,
    model_name: str = "sam3.pt",
    buffer_px: int = 20,
) -> pd.DataFrame:
    """Compute per-link segmentation coverage ratios for each text prompt.

    For each word in *words*, SAM3 segments the aerial photo by that text
    prompt and computes the fraction of pixels inside each link's buffer zone
    that are classified as that word.

    Memory design: link buffer masks are rasterised one at a time within their
    bounding-box crop (~KB each) rather than as full-image arrays (~29 MB each),
    so peak RAM is O(words × image_size) instead of O(links × image_size).

    Parameters
    ----------
    aerial_photo_path:
        Path to the aerial photo PNG (e.g. ``matsuyama_2010_z18.png``).
    geo_json_path:
        Path to the GeoJSON sidecar with the affine transform.
    words:
        List of text prompts (e.g. ``["road", "sidewalk", "tree"]``).
    links:
        List of ``Link`` objects from the road network, in model-alignment order.
    node_lookup:
        Mapping from ``node_id`` to ``Node`` for coordinate lookup.
    cache_dir:
        Directory for caching segmentation masks and area ratios.
    model_name:
        SAM3 model checkpoint name (requires manual download from HuggingFace).
    buffer_px:
        Buffer radius in pixels around each link segment.

    Returns
    -------
    pd.DataFrame
        Columns: ``link_id``, then one column per word with coverage ratios
        in ``[0, 1]``.  Rows aligned with *links* order.
    """
    aerial_photo_path = Path(aerial_photo_path)
    geo_json_path = Path(geo_json_path)
    cache_dir = Path(cache_dir)

    # Load geo transform
    geo_info: dict[str, Any] = json.loads(geo_json_path.read_text(encoding="utf-8"))
    transform: list[float] = geo_info["transform"]

    # Load image dimensions only (no full decode)
    with Image.open(aerial_photo_path) as img:
        img_w, img_h = img.size
    log.info("Aerial photo: %d × %d px", img_w, img_h)

    feature_data: dict[str, list] = {"link_id": [float(lk.link_id) for lk in links]}

    if not words:
        result_df = pd.DataFrame(feature_data)
        result_df["link_id"] = result_df["link_id"].astype(int)
        return result_df

    # Partition words into cached (ratios ready) vs uncached (need computation)
    cached_words: list[str] = []
    uncached_words: list[str] = []
    for word in words:
        ratios_path = cache_dir / word / "area_ratios.parquet"
        if ratios_path.exists():
            cached_words.append(word)
        else:
            uncached_words.append(word)

    # Fast path: load cached ratios directly
    for word in cached_words:
        log.info("Loading cached area ratios for word=%r", word)
        cached_df = pd.read_parquet(cache_dir / word / "area_ratios.parquet").set_index("link_id")
        feature_data[word] = [
            float(cached_df["ratio"].get(lk.link_id, 0.0)) for lk in links
        ]

    if not uncached_words:
        result_df = pd.DataFrame(feature_data)
        result_df["link_id"] = result_df["link_id"].astype(int)
        return result_df

    # Load or run SAM3 segmentation masks for uncached words
    # Peak memory: len(uncached_words) × ~29 MB — feasible on 8 GB machines
    seg_masks: list[np.ndarray] = []
    for word in uncached_words:
        word_cache = cache_dir / word
        mask_path = word_cache / "mask.npy"
        if mask_path.exists():
            log.info("Loading cached mask for word=%r", word)
            seg_mask = np.load(mask_path)
        else:
            word_cache.mkdir(parents=True, exist_ok=True)
            seg_mask = _run_sam3(aerial_photo_path, word, model_name, img_h, img_w)
            np.save(mask_path, seg_mask)
            log.info(
                "Saved mask for word=%r (coverage=%.2f%%)",
                word,
                100.0 * seg_mask.mean(),
            )
        seg_masks.append(seg_mask)

    # Compute per-link ratios one link at a time using bounding-box crops
    # Buffer mask memory per link: O(link_length × buffer_px) instead of O(H × W)
    log.info(
        "Computing coverage ratios for %d links × %d words (buffer_px=%d) …",
        len(links),
        len(uncached_words),
        buffer_px,
    )
    ratios_by_word: list[list[float]] = [[] for _ in uncached_words]
    for lk in links:
        overlaps = _compute_link_overlaps_cropped(
            lk, node_lookup, transform, img_h, img_w, buffer_px, seg_masks
        )
        for i, ratio in enumerate(overlaps):
            ratios_by_word[i].append(ratio)

    # Cache ratios and populate feature_data
    for word, ratios in zip(uncached_words, ratios_by_word):
        word_cache = cache_dir / word
        word_cache.mkdir(parents=True, exist_ok=True)
        ratio_df = pd.DataFrame({
            "link_id": [lk.link_id for lk in links],
            "ratio": ratios,
        })
        ratio_df.to_parquet(word_cache / "area_ratios.parquet", index=False)
        log.info("Saved area ratios for word=%r", word)
        feature_data[word] = ratios

    # Restore original word order in output columns
    result_df = pd.DataFrame({"link_id": feature_data["link_id"]})
    for word in words:
        result_df[word] = feature_data[word]
    result_df["link_id"] = result_df["link_id"].astype(int)
    return result_df


def visualize_sam3_mask(
    aerial_photo_path: str | Path,
    mask: np.ndarray,
    word: str,
    alpha: float = 0.4,
) -> "Image.Image":
    """Overlay a SAM3 segmentation mask on the aerial photo.

    Parameters
    ----------
    aerial_photo_path:
        Path to the aerial photo PNG.
    mask:
        (H, W) bool array from :func:`get_sam3_features` cache.
    word:
        Label used for display.
    alpha:
        Opacity of the overlay (0 = transparent, 1 = opaque).

    Returns
    -------
    PIL.Image.Image
        Composited RGB image with the mask highlighted in red.
    """
    base = Image.open(aerial_photo_path).convert("RGBA")
    img_w, img_h = base.size

    # Build overlay via numpy array assignment — O(H×W) not O(masked pixels)
    overlay_arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    overlay_arr[mask, 0] = 255
    overlay_arr[mask, 3] = int(255 * alpha)
    overlay = Image.fromarray(overlay_arr, mode="RGBA")

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return composited
