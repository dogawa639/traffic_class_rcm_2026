"""Visualisation utilities for route-choice models using folium."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rcm.entities import ObservedTrip, RoadNetwork

try:
    import folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False
    folium = None  # type: ignore[assignment]


def _check_folium() -> None:
    if not _FOLIUM_AVAILABLE:
        raise ImportError("folium is required: pip install folium")


def visualize_network(
    network: RoadNetwork,
    trips: list[ObservedTrip] | None = None,
    zoom_start: int = 15,
) -> "folium.Map":
    """Visualise the road network on an interactive folium map.

    Links are coloured by pedestrian accessibility:
    - **Green**: pedestrian links (``ped=True``)
    - **Gray**: vehicle-only links (``ped=False``)

    When *trips* is provided, trip origins are marked with blue circles and
    destinations with red circles.

    Parameters
    ----------
    network:
        Road network to visualise.
    trips:
        Optional list of observed trips to overlay origin/destination markers.
    zoom_start:
        Initial zoom level for the folium map.

    Returns
    -------
    folium.Map
    """
    _check_folium()

    center_lat = float(np.mean([n.lat for n in network.nodes]))
    center_lon = float(np.mean([n.lon for n in network.nodes]))

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    node_lookup = {n.node_id: n for n in network.nodes}

    for lk in network.links:
        from_node = node_lookup.get(lk.from_node)
        to_node = node_lookup.get(lk.to_node)
        if from_node is None or to_node is None:
            continue
        coords = [
            (from_node.lat, from_node.lon),
            (to_node.lat, to_node.lon),
        ]
        color = "green" if lk.ped else "gray"
        weight = 2 if lk.ped else 1
        folium.PolyLine(  # type: ignore[no-untyped-call]
            coords,
            color=color,
            weight=weight,
            opacity=0.7,
            tooltip=f"link_id={lk.link_id}  len={lk.length_m:.1f}m",
        ).add_to(fmap)

    if trips is not None:
        for trip in trips:
            origin_node = node_lookup.get(trip.origin)
            dest_node = node_lookup.get(trip.destination)
            if origin_node is not None:
                folium.CircleMarker(  # type: ignore[no-untyped-call]
                    location=[origin_node.lat, origin_node.lon],
                    radius=4,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.6,
                    tooltip=f"origin  trip={trip.trip_id}",
                ).add_to(fmap)
            if dest_node is not None:
                folium.CircleMarker(  # type: ignore[no-untyped-call]
                    location=[dest_node.lat, dest_node.lon],
                    radius=4,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.6,
                    tooltip=f"dest  trip={trip.trip_id}",
                ).add_to(fmap)

    return fmap


def visualize_route(
    trip: ObservedTrip,
    network: RoadNetwork,
    predicted_link_ids: list[int] | None = None,
    zoom_start: int = 16,
) -> "folium.Map":
    """Visualise a single trip's actual route and optionally a predicted route.

    - Actual route: green solid polyline
    - Predicted route: red dashed polyline (if provided)

    Parameters
    ----------
    trip:
        The observed trip to visualise.
    network:
        Road network.
    predicted_link_ids:
        Optional list of predicted link IDs to overlay.
    zoom_start:
        Initial zoom level.
    """
    _check_folium()

    node_lookup = {n.node_id: n for n in network.nodes}
    link_lookup = {lk.link_id: lk for lk in network.links}

    def _link_coords(link_ids: list[int]) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for lid in link_ids:
            lk = link_lookup.get(lid)
            if lk is None:
                continue
            fn = node_lookup.get(lk.from_node)
            tn = node_lookup.get(lk.to_node)
            if fn and tn:
                coords.extend([(fn.lat, fn.lon), (tn.lat, tn.lon)])
        return coords

    actual_coords = _link_coords(trip.chosen_route.link_ids)
    if actual_coords:
        center = actual_coords[0]
    else:
        center = (
            float(np.mean([n.lat for n in network.nodes])),
            float(np.mean([n.lon for n in network.nodes])),
        )

    fmap = folium.Map(location=list(center), zoom_start=zoom_start)

    if actual_coords:
        folium.PolyLine(  # type: ignore[no-untyped-call]
            actual_coords,
            color="green",
            weight=4,
            tooltip=f"Actual route  trip={trip.trip_id}",
        ).add_to(fmap)

    if predicted_link_ids is not None:
        pred_coords = _link_coords(predicted_link_ids)
        if pred_coords:
            folium.PolyLine(  # type: ignore[no-untyped-call]
                pred_coords,
                color="red",
                weight=3,
                dash_array="5 10",
                tooltip=f"Predicted route  trip={trip.trip_id}",
            ).add_to(fmap)

    return fmap


def save_html(map_obj: "folium.Map", path: str | Path) -> None:
    """Save a folium Map to an HTML file, creating parent directories if needed."""
    _check_folium()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(path))
