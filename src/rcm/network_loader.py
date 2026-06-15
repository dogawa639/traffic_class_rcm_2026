"""Load road networks from CSV files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from rcm.entities import Link, Node, RoadNetwork

_OPTIONAL_COLS: set[str] = {
    "speed_kmh", "lanes", "sidewalk_width_m",
    "ped_width", "ped_velocity", "veh_velocity",
}
_INT_COLS: set[str] = {"lanes"}


def _parse_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() in {"TRUE", "YES", "1", "T"}
    return bool(val)


def _normalize_node_df(df: pd.DataFrame) -> pd.DataFrame:
    renames: dict[str, str] = {}
    if "node_id" not in df.columns and "id" in df.columns:
        renames["id"] = "node_id"
    return df.rename(columns=renames) if renames else df


def _geodesic_length(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate geodesic distance in metres using the haversine formula."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_link_df(df: pd.DataFrame) -> pd.DataFrame:
    renames: dict[str, str] = {}
    if "link_id" not in df.columns and "id" in df.columns:
        renames["id"] = "link_id"
    if "from_node" not in df.columns and "start" in df.columns:
        renames["start"] = "from_node"
    if "to_node" not in df.columns and "end" in df.columns:
        renames["end"] = "to_node"
    if renames:
        df = df.rename(columns=renames)
    if "length_m" not in df.columns:
        coord_cols = {"start_lon", "start_lat", "end_lon", "end_lat"}
        if coord_cols.issubset(df.columns):
            df["length_m"] = df.apply(
                lambda r: _geodesic_length(r.start_lon, r.start_lat, r.end_lon, r.end_lat),
                axis=1,
            )
        else:
            raise KeyError("length_m column missing and no coordinate columns to compute it from")
    return df


def load_road_network(
    link_csv: str | Path,
    node_csv: str | Path,
    *,
    link_attr_csv: str | Path | None = None,
) -> RoadNetwork:
    """Load a road network from CSV files.

    Parameters
    ----------
    link_csv:
        Path to the link CSV.  Supports columns ``id, start, end`` (auto-renamed
        to ``link_id, from_node, to_node``).  ``length_m`` is computed from
        ``start_lon, start_lat, end_lon, end_lat`` if not present.
    node_csv:
        Path to the node CSV.  Supports ``id`` column (auto-renamed to ``node_id``).
    link_attr_csv:
        Optional path to an attribute CSV (e.g. ``link_attr.csv``) with additional
        columns such as ``lanes, ped_width, ped_velocity, veh_velocity``.
        Merged on ``link_id`` (also handles ``LinkID`` and ``id`` column names).

    Returns
    -------
    RoadNetwork
    """
    node_df = pd.read_csv(Path(node_csv))
    link_df = pd.read_csv(Path(link_csv))

    node_df = _normalize_node_df(node_df)
    link_df = _normalize_link_df(link_df)

    if link_attr_csv is not None:
        attr_df = pd.read_csv(Path(link_attr_csv))
        for raw_col in ("LinkID", "id"):
            if raw_col in attr_df.columns and "link_id" not in attr_df.columns:
                attr_df = attr_df.rename(columns={raw_col: "link_id"})
                break
        link_df = link_df.merge(attr_df, on="link_id", how="left")

    nodes: list[Node] = [
        Node(
            node_id=int(row["node_id"]),
            lon=float(row["lon"]),
            lat=float(row["lat"]),
        )
        for _, row in node_df.iterrows()
    ]

    present_optional = _OPTIONAL_COLS & set(link_df.columns)

    links: list[Link] = []
    for _, row in link_df.iterrows():
        kwargs: dict[str, Any] = {
            "link_id": int(row["link_id"]),
            "from_node": int(row["from_node"]),
            "to_node": int(row["to_node"]),
            "length_m": float(row["length_m"]),
            "car": _parse_bool(row["car"]),
            "ped": _parse_bool(row["ped"]),
        }
        for col in present_optional:
            val = row[col]
            if pd.notna(val):
                kwargs[col] = int(val) if col in _INT_COLS else float(val)
        links.append(Link(**kwargs))

    return RoadNetwork(nodes=nodes, links=links)
