from __future__ import annotations

from dataclasses import dataclass, field

from .body import Body
from .leg import Leg


@dataclass
class Quadruped:
    body: Body
    legs: list[Leg] = field(default_factory=list)

    @classmethod
    def create_kt2_style(
        cls,
        *,
        body_length_m: float = 0.28,
        body_width_m: float = 0.12,
        body_height_m: float = 0.02,
        body_mass_kg: float = 2.4,
        body_elastic_deformation_m: float = 0.002,
        leg_length_m: float = 0.16,
        leg_mass_kg: float = 0.6,
        leg_radius_m: float = 0.010,
        foot_radius_m: float = 0.010,
        foot_static_friction: float = 0.9,
        foot_kinetic_friction: float = 0.65,
    ) -> "Quadruped":
        body = Body(
            length_m=body_length_m,
            width_m=body_width_m,
            height_m=body_height_m,
            mass_kg=body_mass_kg,
            elastic_deformation_m=body_elastic_deformation_m,
        )

        half_length, half_width, _ = body.half_extents()
        mount_z = 0.0

        legs = [
            Leg("front_left",  leg_length_m, leg_mass_kg, foot_static_friction, foot_kinetic_friction, ( half_length,  half_width, mount_z), leg_radius_m=leg_radius_m, foot_radius_m=foot_radius_m),
            Leg("front_right", leg_length_m, leg_mass_kg, foot_static_friction, foot_kinetic_friction, ( half_length, -half_width, mount_z), leg_radius_m=leg_radius_m, foot_radius_m=foot_radius_m),
            Leg("rear_left",   leg_length_m, leg_mass_kg, foot_static_friction, foot_kinetic_friction, (-half_length,  half_width, mount_z), leg_radius_m=leg_radius_m, foot_radius_m=foot_radius_m),
            Leg("rear_right",  leg_length_m, leg_mass_kg, foot_static_friction, foot_kinetic_friction, (-half_length, -half_width, mount_z), leg_radius_m=leg_radius_m, foot_radius_m=foot_radius_m),
        ]
        return cls(body=body, legs=legs)

    @property
    def total_mass_kg(self) -> float:
        return self.body.mass_kg + sum(leg.mass_kg for leg in self.legs)

    def set_leg_motor_velocity(self, leg_name: str, velocity_rad_s: float) -> None:
        self.leg_by_name(leg_name).motor.set_velocity(velocity_rad_s)

    def leg_by_name(self, leg_name: str) -> Leg:
        for leg in self.legs:
            if leg.name == leg_name:
                return leg
        raise ValueError(f"Unknown leg: {leg_name}")
