"""Domain entities for the route-choice model (pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Node(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: int
    lon: float
    lat: float

    @field_validator("node_id")
    @classmethod
    def node_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("node_id must be > 0")
        return v

    @field_validator("lon")
    @classmethod
    def lon_range(cls, v: float) -> float:
        if not -180.0 <= v <= 180.0:
            raise ValueError("lon must be in [-180, 180]")
        return v

    @field_validator("lat")
    @classmethod
    def lat_range(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError("lat must be in [-90, 90]")
        return v


class Link(BaseModel):
    model_config = ConfigDict(frozen=True)

    link_id: int
    from_node: int
    to_node: int
    length_m: float
    car: bool
    ped: bool
    speed_kmh: float | None = None
    lanes: int | None = None
    sidewalk_width_m: float | None = None
    ped_width: float | None = None
    ped_velocity: float | None = None
    veh_velocity: float | None = None

    @field_validator("link_id", "from_node", "to_node")
    @classmethod
    def ids_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ID must be > 0")
        return v

    @field_validator("length_m")
    @classmethod
    def length_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("length_m must be > 0")
        return v

    @model_validator(mode="after")
    def no_self_loop(self) -> Link:
        if self.from_node == self.to_node:
            raise ValueError("self-loop: from_node == to_node")
        return self


class RoadNetwork(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: list[Node]
    links: list[Link]

    @model_validator(mode="after")
    def validate_network(self) -> RoadNetwork:
        if len(self.nodes) < 2:
            raise ValueError("RoadNetwork must have at least 2 nodes")
        node_ids = [n.node_id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node_id in nodes")
        link_ids = [lk.link_id for lk in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("Duplicate link_id in links")
        return self

    def get_node(self, node_id: int) -> Node:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(f"node_id={node_id} not found")

    def get_link(self, link_id: int) -> Link:
        for lk in self.links:
            if lk.link_id == link_id:
                return lk
        raise KeyError(f"link_id={link_id} not found")

    def get_adjacent_links(self, node_id: int) -> list[Link]:
        return [lk for lk in self.links if lk.from_node == node_id]


class Route(BaseModel):
    model_config = ConfigDict(frozen=True)

    trip_id: str
    link_ids: list[int]
    confidence: float
    is_valid: bool

    @field_validator("link_ids")
    @classmethod
    def link_ids_nonempty(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("link_ids must not be empty")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return v


class ObservedTrip(BaseModel):
    model_config = ConfigDict(frozen=True)

    trip_id: str
    origin: int
    destination: int
    chosen_route: Route
    mode: Literal["car", "ped"]

    @model_validator(mode="after")
    def validate_trip(self) -> ObservedTrip:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if not self.chosen_route.is_valid:
            raise ValueError("chosen_route.is_valid must be True")
        return self
