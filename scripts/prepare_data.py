"""One-time data preparation script.

Copies data files from the source image-link-rcm repository.
Land-use features are taken from ``data/processed/link_features.parquet``
(min-max normalized, n_categories=20, EPSG:2446 buffer 50 m) and combined
with ``lanes`` / ``ped_width`` from ``link.csv``.  Always-zero landuse columns
are dropped automatically.

Run from the repository root:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

SRC = Path(__file__).parent.parent.parent / "image-link-rcm"
DST = Path(__file__).parent.parent

# tochi_CD → 土地利用名（国土数値情報 H20）
LANDUSE_NAMES: dict[int, str] = {
    1: "田",
    2: "畑",
    5: "山林",
    6: "水面",
    7: "その他の自然地",
    9: "住宅用地",
    10: "商業用地",
    11: "工業用地",
    12: "公共施設用地",
    13: "公共施設用地（学校）",
    14: "公共施設用地（神社・寺）",
    16: "交通施設用地",
    17: "公共空地",
    19: "その他の空地",
}


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
        src_feat = SRC / "data/processed/link_features.parquet"
        if not src_feat.exists():
            raise FileNotFoundError(
                f"{src_feat} が見つかりません。\n"
                "image-link-rcm の前処理を先に実行してください:\n"
                "  cd ../image-link-rcm && python -m rcm.preprocessing.run"
            )
        df = pd.read_parquet(src_feat)

        # 常にゼロの landuse 列を除外（tochi_CD に対応するコードが存在しない列）
        lc_all = [c for c in df.columns if c.startswith("landuse_")]
        zero_cols = [c for c in lc_all if df[c].max() == 0.0]
        df = df.drop(columns=zero_cols)
        active_lc = [c for c in lc_all if c not in zero_cols]

        dst_feat.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_feat, index=False)

        print(f"  [conv]  link_features.parquet  →  data/dlink_features.parquet")
        print(f"          {len(df)} rows, {len(df.columns)} cols")
        print(f"          除外した常ゼロ列: {zero_cols}")
        print("          使用する土地利用列:")
        for c in active_lc:
            k = int(c.split("_")[1])
            name = LANDUSE_NAMES.get(k, "（不明）")
            print(f"            {c}: tochi_CD={k:02d} {name}")

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
