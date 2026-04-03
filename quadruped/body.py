"""Body-level domain model for the quadruped."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodySpec:
    length_m: float
    width_m: float
    height_m: float
    mass_kg: float
    contact_friction: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("length_m", self.length_m),
            ("width_m", self.width_m),
            ("height_m", self.height_m),
            ("mass_kg", self.mass_kg),
        ):
            if value <= 0.0:
                raise ValueError(f"Body {field_name} must be > 0.")
        if self.contact_friction < 0.0:
            raise ValueError("Body contact_friction must be >= 0.")

    @property
    def half_extents_m(self) -> tuple[float, float, float]:
        return (self.length_m / 2.0, self.width_m / 2.0, self.height_m / 2.0)

    @property
    def principal_inertia(self) -> tuple[float, float, float]:
        length_sq = self.length_m * self.length_m
        width_sq = self.width_m * self.width_m
        height_sq = self.height_m * self.height_m
        return (
            (self.mass_kg / 12.0) * (width_sq + height_sq),
            (self.mass_kg / 12.0) * (length_sq + height_sq),
            (self.mass_kg / 12.0) * (length_sq + width_sq),
        )

    @property
    def corners_body(self) -> tuple[tuple[float, float, float], ...]:
        hx, hy, hz = self.half_extents_m
        corners: list[tuple[float, float, float]] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append((sx * hx, sy * hy, sz * hz))
        return tuple(corners)
