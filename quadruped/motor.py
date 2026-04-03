"""Motor-level domain model for the quadruped."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorSpec:
    control_scale_rad_s: float
    max_velocity_rad_s: float
    max_angular_acceleration_rad_s2: float
    viscous_damping_per_s: float
    velocity_filter_tau_s: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("control_scale_rad_s", self.control_scale_rad_s),
            ("max_velocity_rad_s", self.max_velocity_rad_s),
            ("max_angular_acceleration_rad_s2", self.max_angular_acceleration_rad_s2),
            ("viscous_damping_per_s", self.viscous_damping_per_s),
            ("velocity_filter_tau_s", self.velocity_filter_tau_s),
        ):
            if value <= 0.0:
                raise ValueError(f"Motor {field_name} must be > 0.")
