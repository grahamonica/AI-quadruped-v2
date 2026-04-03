"""Leg-level domain model for the quadruped."""

from __future__ import annotations

from dataclasses import dataclass


LEG_ROTATION_AXIS_BODY = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class LegSpec:
    name: str
    mount_point_body: tuple[float, float, float]
    length_m: float
    mass_kg: float
    radius_m: float
    foot_radius_m: float
    elastic_deformation_m: float
    static_friction: float
    kinetic_friction: float
    body_contact_samples: int
    rotation_axis_body: tuple[float, float, float] = LEG_ROTATION_AXIS_BODY

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Leg name must be non-empty.")
        for field_name, value in (
            ("length_m", self.length_m),
            ("mass_kg", self.mass_kg),
            ("radius_m", self.radius_m),
            ("foot_radius_m", self.foot_radius_m),
        ):
            if value <= 0.0:
                raise ValueError(f"Leg {field_name} must be > 0.")
        if self.elastic_deformation_m < 0.0:
            raise ValueError("Leg elastic_deformation_m must be >= 0.")
        if self.static_friction < 0.0 or self.kinetic_friction < 0.0:
            raise ValueError("Leg friction values must be >= 0.")
        if self.kinetic_friction > self.static_friction:
            raise ValueError("Leg kinetic friction must not exceed static friction.")
        if self.body_contact_samples <= 0:
            raise ValueError("Leg body_contact_samples must be > 0.")

    @property
    def inertia_about_mount(self) -> float:
        return self.mass_kg * (self.length_m**2) / 3.0

    @property
    def body_sample_fractions(self) -> tuple[float, ...]:
        step = 1.0 / float(self.body_contact_samples + 1)
        return tuple((index + 1) * step for index in range(self.body_contact_samples))
