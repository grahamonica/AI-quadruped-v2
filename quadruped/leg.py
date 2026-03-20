from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin

from .imu import IMU
from .motor import Motor


@dataclass
class Leg:
    name: str
    length_m: float
    mass_kg: float
    foot_static_friction: float
    foot_kinetic_friction: float
    mount_point_xyz_m: tuple[float, float, float]
    leg_radius_m: float = 0.010
    foot_radius_m: float = 0.015
    imu: IMU = field(default_factory=IMU)
    motor: Motor = field(default_factory=Motor)
    angle_rad: float = 0.0
    angular_velocity_rad_s: float = 0.0
    angular_acceleration_rad_s2: float = 0.0

    def advance_motor(self, dt_s: float) -> None:
        self.angular_acceleration_rad_s2 = self.motor.step(dt_s)
        self.angular_velocity_rad_s = self.motor.current_velocity_rad_s
        self.angle_rad += self.angular_velocity_rad_s * dt_s

    def foot_offset_from_mount_m(self) -> tuple[float, float, float]:
        return (
            self.length_m * sin(self.angle_rad),
            0.0,
            -self.length_m * cos(self.angle_rad),
        )

    def foot_velocity_from_mount_m_s(self) -> tuple[float, float, float]:
        return (
            self.length_m * cos(self.angle_rad) * self.angular_velocity_rad_s,
            0.0,
            self.length_m * sin(self.angle_rad) * self.angular_velocity_rad_s,
        )

    def foot_acceleration_from_mount_m_s2(self) -> tuple[float, float, float]:
        return (
            (-self.length_m * sin(self.angle_rad) * (self.angular_velocity_rad_s**2))
            + (self.length_m * cos(self.angle_rad) * self.angular_acceleration_rad_s2),
            0.0,
            (self.length_m * cos(self.angle_rad) * (self.angular_velocity_rad_s**2))
            + (self.length_m * sin(self.angle_rad) * self.angular_acceleration_rad_s2),
        )

    def com_offset_from_mount_m(self) -> tuple[float, float, float]:
        half_length = self.length_m / 2.0
        return (
            half_length * sin(self.angle_rad),
            0.0,
            -half_length * cos(self.angle_rad),
        )

    def com_acceleration_from_mount_m_s2(self) -> tuple[float, float, float]:
        """Linear acceleration of the leg's centre of mass relative to its mount point."""
        half_length = self.length_m / 2.0
        return (
            (-half_length * sin(self.angle_rad) * (self.angular_velocity_rad_s**2))
            + (half_length * cos(self.angle_rad) * self.angular_acceleration_rad_s2),
            0.0,
            (half_length * cos(self.angle_rad) * (self.angular_velocity_rad_s**2))
            + (half_length * sin(self.angle_rad) * self.angular_acceleration_rad_s2),
        )
