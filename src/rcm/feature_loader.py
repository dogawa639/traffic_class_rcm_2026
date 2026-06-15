"""Utility for loading and selecting per-link features from dlink_features.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_link_features(
    dlink_path: str | Path,
    link_ids: list[int],
    dlink_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load per-link feature matrix aligned with *link_ids*.

    Parameters
    ----------
    dlink_path:
        Path to ``dlink_features.parquet``.
    link_ids:
        Ordered list of link IDs matching ``network.links``.
    dlink_cols:
        Columns to use from the parquet file.  ``None`` uses all columns
        except ``link_id``.

    Returns
    -------
    X : ndarray of shape (n_links, n_features)
        Feature matrix, zero-filled for links absent in the parquet file.
    feature_names : list[str]
        Column names corresponding to ``X`` columns.
    """
    df = pd.read_parquet(Path(dlink_path))
    all_cols = [c for c in df.columns if c != "link_id"]

    if dlink_cols is None:
        feature_names = all_cols
    else:
        missing = [c for c in dlink_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"config の features.dlink_cols に存在しない列が指定されています: {missing}\n"
                f"利用可能な列: {all_cols}"
            )
        feature_names = list(dlink_cols)

    aligned = df.set_index("link_id").reindex(link_ids)[feature_names].fillna(0.0)
    return aligned.values, feature_names
