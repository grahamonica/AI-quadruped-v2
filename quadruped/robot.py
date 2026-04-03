"""Robot-level domain model that aggregates body, legs, and motors."""

from __future__ import annotations

from dataclasses import dataclass

from ai.config import RuntimeSpec

from .body import BodySpec
from .leg import LegSpec
from .motor import MotorSpec


LEG_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


@dataclass(frozen=True)
class QuadrupedRobot:
    body: BodySpec
    motor: MotorSpec
    legs: tuple[LegSpec, ...]

    def __post_init__(self) -> None:
        if len(self.legs) != 4:
            raise ValueError("QuadrupedRobot requires exactly 4 legs.")

    @classmethod
    def from_runtime_spec(cls, spec: RuntimeSpec) -> "QuadrupedRobot":
        body = BodySpec(
            length_m=spec.robot.body_length_m,
            width_m=spec.robot.body_width_m,
            height_m=spec.robot.body_height_m,
            mass_kg=spec.robot.body_mass_kg,
            contact_friction=spec.friction.body,
        )
        motor = MotorSpec(
            control_scale_rad_s=spec.robot.motor_scale,
            max_velocity_rad_s=spec.robot.max_motor_rad_s,
            max_angular_acceleration_rad_s2=spec.robot.motor_max_angular_acceleration_rad_s2,
            viscous_damping_per_s=spec.robot.motor_viscous_damping_per_s,
            velocity_filter_tau_s=spec.robot.motor_velocity_filter_tau_s,
        )
        mounts = (
            (body.length_m / 2.0, body.width_m / 2.0, 0.0),
            (body.length_m / 2.0, -body.width_m / 2.0, 0.0),
            (-body.length_m / 2.0, body.width_m / 2.0, 0.0),
            (-body.length_m / 2.0, -body.width_m / 2.0, 0.0),
        )
        legs = tuple(
            LegSpec(
                name=leg_name,
                mount_point_body=mount,
                length_m=spec.robot.leg_length_m,
                mass_kg=spec.robot.leg_mass_kg,
                radius_m=spec.robot.leg_radius_m,
                foot_radius_m=spec.robot.foot_radius_m,
                elastic_deformation_m=spec.robot.elastic_deformation_m,
                static_friction=spec.friction.foot_static,
                kinetic_friction=spec.friction.foot_kinetic,
                body_contact_samples=spec.robot.leg_body_samples,
            )
            for leg_name, mount in zip(LEG_NAMES, mounts, strict=True)
        )
        return cls(body=body, motor=motor, legs=legs)

    @property
    def total_mass_kg(self) -> float:
        return self.body.mass_kg + sum(leg.mass_kg for leg in self.legs)

    @property
    def leg_names(self) -> tuple[str, ...]:
        return tuple(leg.name for leg in self.legs)

    @property
    def mount_points_body(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(leg.mount_point_body for leg in self.legs)

    @property
    def body_corners_body(self) -> tuple[tuple[float, float, float], ...]:
        return self.body.corners_body

    @property
    def leg_inertia_about_mount(self) -> tuple[float, ...]:
        return tuple(leg.inertia_about_mount for leg in self.legs)

    @property
    def leg_body_fractions(self) -> tuple[float, ...]:
        return self.legs[0].body_sample_fractions
