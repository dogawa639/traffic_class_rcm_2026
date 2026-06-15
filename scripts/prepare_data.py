"""One-time data preparation script.

Copies data files from the source image-link-rcm repository and converts
link_attr.csv to dlink_features.parquet.

Run from the repository root:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

SRC = Path(__file__).parent.parent.parent / "image-link-rcm"
DST = Path(__file__).parent.parent


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  [skip]  {dst.relative_to(DST)} already exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [copy]  {src.name}  →  {dst.relative_to(DST)}")


def copy_dir_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  [skip]  {dst.relative_to(DST)}/ already exists")
        return
    shutil.copytree(src, dst)
    print(f"  [copy]  {src.name}/  →  {dst.relative_to(DST)}/")


def main() -> None:
    print("=== Data preparation ===")
    print(f"Source: {SRC}")
    print(f"Target: {DST}")
    print()

    # --- Network CSVs ---
    print("[1/5] Network CSVs")
    copy_if_missing(
        SRC / "data/raw/network/matsuyama/link.csv",
        DST / "data/network/link.csv",
    )
    copy_if_missing(
        SRC / "data/raw/network/matsuyama/node.csv",
        DST / "data/network/node.csv",
    )

    # --- dlink_features.parquet ---
    print("[2/5] dlink_features.parquet")
    dst_feat = DST / "data/dlink_features.parquet"
    if dst_feat.exists():
        print(f"  [skip]  {dst_feat.relative_to(DST)} already exists")
    else:
        src_attr = SRC / "data/raw/network/matsuyama/link_attr.csv"
        df = pd.read_csv(src_attr)
        df = df.rename(columns={"LinkID": "link_id"})
        dst_feat.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_feat, index=False)
        print(f"  [conv]  link_attr.csv  →  data/dlink_features.parquet  ({len(df)} rows, {len(df.columns)} cols)")

    # --- routes.parquet ---
    print("[3/5] routes.parquet")
    copy_if_missing(
        SRC / "data/interim/2007_matsuyama/ped/map_matched/routes.parquet",
        DST / "data/routes.parquet",
    )

    # --- Aerial photo ---
    print("[4/5] Aerial photo")
    copy_if_missing(
        SRC / "data/raw/aerial_photo/matsuyama/matsuyama_2010_z18.png",
        DST / "data/aerial_photo/matsuyama_2010_z18.png",
    )
    copy_if_missing(
        SRC / "data/raw/aerial_photo/matsuyama/matsuyama_2010_z18_geo.json",
        DST / "data/aerial_photo/matsuyama_2010_z18_geo.json",
    )

    # --- Land use ---
    print("[5/5] Land use data")
    copy_dir_if_missing(
        SRC / "data/raw/land_use/matsuyama",
        DST / "data/land_use/matsuyama",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
