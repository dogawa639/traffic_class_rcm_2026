"""Load map-matched pedestrian trips from parquet files."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from rcm.entities import ObservedTrip, Route

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD: float = 0.5


def load_trips(
    routes_parquet: str | Path,
    *,
    min_confidence: float = CONFIDENCE_THRESHOLD,
    mode: str = "ped",
) -> list[ObservedTrip]:
    """Load map-matched routes and return a list of ObservedTrip objects.

    Parameters
    ----------
    routes_parquet:
        Path to ``routes.parquet``.  Expected columns:
        ``trip_id, link_ids (list[int]), confidence, is_valid, origin, destination``.
    min_confidence:
        Trips with confidence below this threshold are filtered out (default 0.5).
    mode:
        Travel mode label attached to each ObservedTrip (default ``"ped"``).

    Returns
    -------
    list[ObservedTrip]
        Filtered, valid trips sorted by trip_id.
    """
    path = Path(routes_parquet)
    if not path.exists():
        raise FileNotFoundError(f"routes.parquet not found: {path}")

    df = pd.read_parquet(path)
    log.info("Loaded %d rows from %s", len(df), path)

    # Filter to valid, high-confidence routes
    df = df[df["is_valid"] & (df["confidence"] >= min_confidence)].copy()
    log.info("After filtering (is_valid=True, confidence≥%.2f): %d trips", min_confidence, len(df))

    trips: list[ObservedTrip] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            link_ids = [int(x) for x in row["link_ids"]]
            route = Route(
                trip_id=str(row["trip_id"]),
                link_ids=link_ids,
                confidence=float(row["confidence"]),
                is_valid=bool(row["is_valid"]),
            )
            trip = ObservedTrip(
                trip_id=str(row["trip_id"]),
                origin=int(row["origin"]),
                destination=int(row["destination"]),
                chosen_route=route,
                mode=mode,  # type: ignore[arg-type]
            )
            trips.append(trip)
        except Exception as exc:
            skipped += 1
            warnings.warn(f"Skipping trip {row.get('trip_id', '?')}: {exc}", stacklevel=2)

    if skipped:
        log.warning("Skipped %d invalid trips", skipped)
    log.info("Loaded %d valid ObservedTrip objects", len(trips))
    return sorted(trips, key=lambda t: t.trip_id)
