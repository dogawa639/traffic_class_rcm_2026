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


def _build_link_buffer_mask(
    lk: Link,
    node_lookup: dict[int, Node],
    transform: list[float],
    img_h: int,
    img_w: int,
    buffer_px: float,
) -> np.ndarray:
    """Return a (H, W) bool mask for the buffered road link in pixel space.

    The link segment is dilated by *buffer_px* pixels using a Shapely buffer,
    then rasterised with PIL.  Returns an all-False mask if the link falls
    entirely outside the image.
    """
    from_node = node_lookup.get(lk.from_node)
    to_node = node_lookup.get(lk.to_node)
    if from_node is None or to_node is None:
        return np.zeros((img_h, img_w), dtype=bool)

    col1, row1 = _lon_lat_to_pixel(from_node.lon, from_node.lat, transform)
    col2, row2 = _lon_lat_to_pixel(to_node.lon, to_node.lat, transform)

    # Degenerate segment: use a circular buffer around midpoint
    if abs(col1 - col2) < 0.5 and abs(row1 - row2) < 0.5:
        line = LineString([(col1, row1), (col1 + 0.5, row1)])
    else:
        line = LineString([(col1, row1), (col2, row2)])

    buffered = line.buffer(buffer_px)

    # Quick bounds check — skip if outside image
    minx, miny, maxx, maxy = buffered.bounds
    if maxx < 0 or maxy < 0 or minx > img_w or miny > img_h:
        return np.zeros((img_h, img_w), dtype=bool)

    # Rasterise polygon via PIL
    img_mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(img_mask)
    exterior = list(buffered.exterior.coords)
    draw.polygon([(c[0], c[1]) for c in exterior], fill=1)
    return np.array(img_mask, dtype=bool)


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

    # Load image to get dimensions (no full decode needed — just header)
    with Image.open(aerial_photo_path) as img:
        img_w, img_h = img.size
    log.info("Aerial photo: %d × %d px", img_w, img_h)

    if not words:
        return pd.DataFrame({"link_id": [lk.link_id for lk in links]})

    # Precompute link buffer masks (only once per session)
    log.info("Building link buffer masks (buffer_px=%d) …", buffer_px)
    buffer_masks: list[np.ndarray] = [
        _build_link_buffer_mask(lk, node_lookup, transform, img_h, img_w, buffer_px)
        for lk in links
    ]
    buffer_areas = np.array([m.sum() for m in buffer_masks], dtype=float)

    feature_data: dict[str, list[float]] = {
        "link_id": [float(lk.link_id) for lk in links]
    }

    for word in words:
        word_cache = cache_dir / word
        mask_path = word_cache / "mask.npy"
        ratios_path = word_cache / "area_ratios.parquet"

        if ratios_path.exists():
            # Fast path: load cached ratios
            log.info("Loading cached area ratios for word=%r", word)
            cached_df = pd.read_parquet(ratios_path).set_index("link_id")
            feature_data[word] = [
                float(cached_df["ratio"].get(lk.link_id, 0.0)) for lk in links
            ]
            continue

        # Load or compute segmentation mask
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

        # Compute per-link coverage ratios
        ratios: list[float] = []
        for i, buf_mask in enumerate(buffer_masks):
            area = buffer_areas[i]
            if area == 0:
                ratios.append(0.0)
            else:
                overlap = float((seg_mask & buf_mask).sum())
                ratios.append(overlap / area)

        # Cache ratios
        word_cache.mkdir(parents=True, exist_ok=True)
        ratio_df = pd.DataFrame({
            "link_id": [lk.link_id for lk in links],
            "ratio": ratios,
        })
        ratio_df.to_parquet(ratios_path, index=False)
        log.info("Saved area ratios for word=%r", word)

        feature_data[word] = ratios

    result_df = pd.DataFrame(feature_data)
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
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Highlight masked pixels in red
    ys, xs = np.where(mask)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=(255, 0, 0, int(255 * alpha)))

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return composited
