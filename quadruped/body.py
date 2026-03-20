from __future__ import annotations

from dataclasses import dataclass, field

from .imu import IMU


@dataclass
class Body:
    length_m: float
    width_m: float
    height_m: float
    mass_kg: float
    elastic_deformation_m: float = 0.002
    imu: IMU = field(default_factory=IMU)
    position_xyz_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_xyz_m_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration_xyz_m_s2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_xyz_rad: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity_xyz_rad_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_acceleration_xyz_rad_s2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def half_extents(self) -> tuple[float, float, float]:
        return (self.length_m / 2.0, self.width_m / 2.0, self.height_m / 2.0)

    def principal_inertia_kg_m2(self) -> tuple[float, float, float]:
        lx = self.length_m
        wy = self.width_m
        hz = self.height_m
        mass = self.mass_kg
        return (
            (mass / 12.0) * ((wy * wy) + (hz * hz)),
            (mass / 12.0) * ((lx * lx) + (hz * hz)),
            (mass / 12.0) * ((lx * lx) + (wy * wy)),
        )

    def corners_body_frame(self) -> list[tuple[float, float, float]]:
        half_length, half_width, half_height = self.half_extents()
        return [
            (sx * half_length, sy * half_width, sz * half_height)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

    def contact_sample_points_body_frame(self) -> list[tuple[float, float, float]]:
        """Contact sample points representing the slightly elastically deformable body surface.

        The body is modeled as slightly elastically deformable so that resting flat on the
        floor produces a non-negligible contact area rather than point contacts at the four
        bottom corners only.  The bottom face is sampled at its corners, edge midpoints, and
        centroid; the four top corners are included to detect body roll/flip contact.
        """
        half_length, half_width, half_height = self.half_extents()
        bottom_z = -half_height
        return [
            # Bottom face: 4 corners
            (-half_length, -half_width, bottom_z),
            (-half_length,  half_width, bottom_z),
            ( half_length, -half_width, bottom_z),
            ( half_length,  half_width, bottom_z),
            # Bottom face: 4 edge midpoints
            (-half_length,        0.0, bottom_z),
            ( half_length,        0.0, bottom_z),
            (        0.0, -half_width, bottom_z),
            (        0.0,  half_width, bottom_z),
            # Bottom face: centroid
            (        0.0,        0.0, bottom_z),
            # Top face: 4 corners (detect roll/flip contact)
            (-half_length, -half_width,  half_height),
            (-half_length,  half_width,  half_height),
            ( half_length, -half_width,  half_height),
            ( half_length,  half_width,  half_height),
        ]
