from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, cos, sin

from .robot import Quadruped


def _v_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_scale(a: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _v_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _v_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


_LEG_BODY_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75)


def _rotation_matrix_xyz(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple[tuple[float, float, float], ...]:
    cr = cos(roll_rad)
    sr = sin(roll_rad)
    cp = cos(pitch_rad)
    sp = sin(pitch_rad)
    cy = cos(yaw_rad)
    sy = sin(yaw_rad)
    return (
        (cy * cp, (cy * sp * sr) - (sy * cr), (cy * sp * cr) + (sy * sr)),
        (sy * cp, (sy * sp * sr) + (cy * cr), (sy * sp * cr) - (cy * sr)),
        (-sp, cp * sr, cp * cr),
    )


def _mat_vec_mul(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        _v_dot(matrix[0], vector),
        _v_dot(matrix[1], vector),
        _v_dot(matrix[2], vector),
    )


@dataclass
class ContactMemory:
    is_in_contact: bool = False
    tangential_anchor_xy_m: tuple[float, float] = (0.0, 0.0)


@dataclass
class LegForceState:
    leg_name: str
    contact_mode: str = "airborne"
    foot_position_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    foot_velocity_xyz_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_force_xyz_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_reaction_xyz_n: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class QuadrupedEnvironment:
    robot: Quadruped
    gravity_m_s2: float = 9.81
    floor_height_m: float = 0.0
    # Contact spring stiffness. Softened slightly so the overdamped contact
    # coefficients below can fully absorb impact energy without numerical ringing.
    # c_crit_normal  = 2*sqrt(20000*2.5) = 447 N·s/m  → 3500/447 ≈ 7.8× overdamped
    # c_crit_tangent = 2*sqrt(7000*2.5)  = 265 N·s/m  → 450/265  ≈ 1.7× overdamped
    # All damping terms are proper velocity-proportional forces (F = -c·v) in the
    # equations of motion, not post-integration velocity patches.
    normal_stiffness_n_m: float = 20000.0
    normal_damping_n_s_m: float = 3500.0
    tangential_stiffness_n_m: float = 7000.0
    tangential_damping_n_s_m: float = 450.0
    body_contact_friction: float = 0.35
    # High viscous damping on body motion: this is the primary mechanism for
    # damping all oscillation. Any velocity → damping force → motion dies out.
    # Contact forces (which are velocity-dependent) are automatically reduced
    # because the velocity that drives them is kept small.  At 80 N·s/m a 0.1 m/s
    # body velocity produces 8 N of resistance (≈33% of body weight), which
    # kills jitter within a frame or two without needing any ad-hoc velocity patches.
    # Damping when at least one foot/body corner is in contact with the ground.
    # High values kill oscillation; applied implicitly so any coefficient is safe.
    angular_damping_n_m_s: float = 8.0
    linear_damping_n_s_m: float = 80.0
    # Damping when fully airborne.  Must be low so free-fall feels realistic —
    # at 3 N·s/m the terminal velocity is mg/b ≈ 35 N / 3 = 12 m/s, which is
    # well above any drop the robot will experience.  The previous single value
    # (80) limited terminal velocity to 0.44 m/s, making gravity look weak.
    airborne_linear_damping_n_s_m: float = 3.0
    airborne_angular_damping_n_m_s: float = 1.0
    max_contact_force_n: float = 120.0
    max_substep_s: float = 1.0 / 4000.0
    unloading_stiffness_scale: float = 0.4
    sleep_linear_speed_threshold_m_s: float = 0.01
    sleep_angular_speed_threshold_rad_s: float = 0.06
    leg_force_states: dict[str, LegForceState] = field(default_factory=dict)
    body_contact_forces_xyz_n: list[tuple[float, float, float]] = field(default_factory=list)
    time_s: float = 0.0

    def __post_init__(self) -> None:
        self._rest_contact_buffer_m = (
            self.robot.total_mass_kg * self.gravity_m_s2
        ) / (
            max(len(self.robot.legs), 1) * self.normal_stiffness_n_m
        )
        self._contact_memory = {leg.name: ContactMemory() for leg in self.robot.legs}
        self._body_contact_memory = {
            index: ContactMemory() for index, _ in enumerate(self.robot.body.contact_sample_points_body_frame())
        }
        self._leg_body_contact_memory: dict[tuple[str, int], ContactMemory] = {
            (leg.name, i): ContactMemory()
            for leg in self.robot.legs
            for i in range(len(_LEG_BODY_FRACTIONS))
        }
        self.reset_pose()

    def reset_pose(self) -> None:
        body = self.robot.body
        # Lowest contact point is the bottom of each spherical foot tip.
        lowest_foot_body_z = min(
            leg.mount_point_xyz_m[2] + leg.foot_offset_from_mount_m()[2] - leg.foot_radius_m
            for leg in self.robot.legs
        )
        body.position_xyz_m[:] = [0.0, 0.0, self.floor_height_m - lowest_foot_body_z]
        body.velocity_xyz_m_s[:] = [0.0, 0.0, 0.0]
        body.acceleration_xyz_m_s2[:] = [0.0, 0.0, 0.0]
        body.rotation_xyz_rad[:] = [0.0, 0.0, 0.0]
        body.angular_velocity_xyz_rad_s[:] = [0.0, 0.0, 0.0]
        body.angular_acceleration_xyz_rad_s2[:] = [0.0, 0.0, 0.0]

        for leg in self.robot.legs:
            leg.angle_rad = 0.0
            leg.angular_velocity_rad_s = 0.0
            leg.angular_acceleration_rad_s2 = 0.0
            leg.motor.current_velocity_rad_s = 0.0
            leg.motor.target_velocity_rad_s = 0.0
            leg.motor.smoothed_target_rad_s = 0.0
            mount_world = self._body_point_world(leg.mount_point_xyz_m)
            foot_position = _v_add(mount_world, self._body_vector_world(leg.foot_offset_from_mount_m()))
            self._contact_memory[leg.name] = ContactMemory(
                is_in_contact=True,
                tangential_anchor_xy_m=(foot_position[0], foot_position[1]),
            )
            self.leg_force_states[leg.name] = LegForceState(leg_name=leg.name)

        floor_threshold = self.floor_height_m + body.elastic_deformation_m + 1e-5
        for index, sample_point in enumerate(body.contact_sample_points_body_frame()):
            point = self._body_point_world(sample_point)
            self._body_contact_memory[index] = ContactMemory(
                is_in_contact=point[2] <= floor_threshold,
                tangential_anchor_xy_m=(point[0], point[1]),
            )

        for leg in self.robot.legs:
            for i in range(len(_LEG_BODY_FRACTIONS)):
                self._leg_body_contact_memory[(leg.name, i)] = ContactMemory()

        self.time_s = 0.0
        leg_kinematics = self._compute_leg_kinematics()
        leg_contacts = self._compute_leg_contacts_from_kinematics(leg_kinematics)
        body_contacts = self._compute_body_contacts()
        self.body_contact_forces_xyz_n = [contact["force"] for contact in body_contacts]
        self._update_imu_and_force_state(leg_contacts, body_contacts)

    def advance(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return

        substeps = max(1, ceil(dt_s / self.max_substep_s))
        substep_dt_s = dt_s / substeps
        for _ in range(substeps):
            self._advance_substep(substep_dt_s)
            self.time_s += substep_dt_s

    def _advance_substep(self, dt_s: float) -> None:
        for leg in self.robot.legs:
            leg.advance_motor(dt_s)

        leg_kinematics = self._compute_leg_kinematics()
        leg_contacts = self._compute_leg_contacts_from_kinematics(leg_kinematics)
        body_contacts = self._compute_body_contacts()
        leg_body_contacts = self._compute_leg_body_contacts()

        body = self.robot.body
        total_force_n = (0.0, 0.0, -self.robot.total_mass_kg * self.gravity_m_s2)
        # Damping is applied implicitly after integration (see below) so it is
        # NOT included as an explicit force here.  Explicit -b*v with high b causes
        # b*dt/m > 2, which violates the Euler stability criterion and produces the
        # numerical explosion seen as OverflowError.
        total_torque_n_m = (0.0, 0.0, 0.0)

        # Newton's 3rd law: leg inertia creates reaction forces/torques on body.
        # When a leg accelerates (motor or contact-driven), the body feels the
        # equal-and-opposite force.  Omitting this causes undamped jitter because
        # the body dynamics are blind to leg motion.
        leg_rot_axis_body = (0.0, 1.0, 0.0)  # legs swing in the body x-z plane
        leg_rot_axis_world = self._body_vector_world(leg_rot_axis_body)
        for leg in self.robot.legs:
            # Linear inertia reaction: F_reaction = -m_leg * a_leg_com_relative
            com_accel_body = leg.com_acceleration_from_mount_m_s2()
            com_accel_world = self._body_vector_world(com_accel_body)
            inertia_force = _v_scale(com_accel_world, -leg.mass_kg)
            total_force_n = _v_add(total_force_n, inertia_force)

            # Rotational inertia reaction: τ = -I_leg * α_leg (about swing axis)
            # Moment of inertia of cylinder about one end: I = m*L²/3
            # (spherical foot tip contribution ~m*(2r²/5 + L²) negligible since foot_radius_m << length_m)
            inertia_about_mount_kg_m2 = leg.mass_kg * leg.length_m**2 / 3.0
            rot_reaction = _v_scale(leg_rot_axis_world, -inertia_about_mount_kg_m2 * leg.angular_acceleration_rad_s2)
            total_torque_n_m = _v_add(total_torque_n_m, rot_reaction)

            # Torque from the linear inertia force acting at the mount point
            r_mount_world = self._body_vector_world(leg.mount_point_xyz_m)
            total_torque_n_m = _v_add(total_torque_n_m, _v_cross(r_mount_world, inertia_force))

        for contact in leg_contacts.values():
            total_force_n = _v_add(total_force_n, contact["force"])
            total_torque_n_m = _v_add(total_torque_n_m, _v_cross(contact["r_world"], contact["force"]))

        for contact in body_contacts:
            total_force_n = _v_add(total_force_n, contact["force"])
            total_torque_n_m = _v_add(total_torque_n_m, _v_cross(contact["r_world"], contact["force"]))

        for contact in leg_body_contacts:
            total_force_n = _v_add(total_force_n, contact["force"])
            total_torque_n_m = _v_add(total_torque_n_m, _v_cross(contact["r_world"], contact["force"]))

        total_mass = self.robot.total_mass_kg
        prev_velocity = tuple(body.velocity_xyz_m_s)
        prev_angular_velocity = tuple(body.angular_velocity_xyz_rad_s)

        # Choose damping based on whether any contact point is currently active.
        # Airborne: near-zero damping so the robot free-falls at realistic speed.
        # Grounded: high damping to absorb contact oscillations.
        any_grounded = (
            any(contact["force"][2] > 0.0 for contact in leg_contacts.values())
            or any(contact["force"][2] > 0.0 for contact in body_contacts)
            or any(contact["force"][2] > 0.0 for contact in leg_body_contacts)
        )
        active_linear_damping = self.linear_damping_n_s_m if any_grounded else self.airborne_linear_damping_n_s_m
        active_angular_damping = self.angular_damping_n_m_s if any_grounded else self.airborne_angular_damping_n_m_s

        linear_acceleration_undamped = _v_scale(total_force_n, 1.0 / total_mass)
        # Implicit viscous damping: solve m*v' = F - b*v implicitly.
        # v_new = (v_old + a*dt) / (1 + b*dt/m).  Unconditionally stable for any b.
        linear_damp_divisor = 1.0 + active_linear_damping * dt_s / total_mass
        body.velocity_xyz_m_s[:] = [
            (body.velocity_xyz_m_s[i] + linear_acceleration_undamped[i] * dt_s) / linear_damp_divisor
            for i in range(3)
        ]
        body.acceleration_xyz_m_s2[:] = [
            (body.velocity_xyz_m_s[i] - prev_velocity[i]) / dt_s for i in range(3)
        ]
        body.position_xyz_m[:] = [
            body.position_xyz_m[0] + (body.velocity_xyz_m_s[0] * dt_s),
            body.position_xyz_m[1] + (body.velocity_xyz_m_s[1] * dt_s),
            body.position_xyz_m[2] + (body.velocity_xyz_m_s[2] * dt_s),
        ]

        inertia_x, inertia_y, inertia_z = body.principal_inertia_kg_m2()
        angular_acceleration_undamped = (
            total_torque_n_m[0] / max(inertia_x, 1e-6),
            total_torque_n_m[1] / max(inertia_y, 1e-6),
            total_torque_n_m[2] / max(inertia_z, 1e-6),
        )
        # Implicit angular damping: I*ω' = τ - c*ω  →  ω_new = (ω_old + α*dt) / (1 + c*dt/I)
        ang_dt = active_angular_damping * dt_s
        body.angular_velocity_xyz_rad_s[:] = [
            (body.angular_velocity_xyz_rad_s[0] + angular_acceleration_undamped[0] * dt_s) / (1.0 + ang_dt / max(inertia_x, 1e-6)),
            (body.angular_velocity_xyz_rad_s[1] + angular_acceleration_undamped[1] * dt_s) / (1.0 + ang_dt / max(inertia_y, 1e-6)),
            (body.angular_velocity_xyz_rad_s[2] + angular_acceleration_undamped[2] * dt_s) / (1.0 + ang_dt / max(inertia_z, 1e-6)),
        ]
        body.angular_acceleration_xyz_rad_s2[:] = [
            (body.angular_velocity_xyz_rad_s[i] - prev_angular_velocity[i]) / dt_s for i in range(3)
        ]
        body.rotation_xyz_rad[:] = [
            body.rotation_xyz_rad[0] + (body.angular_velocity_xyz_rad_s[0] * dt_s),
            body.rotation_xyz_rad[1] + (body.angular_velocity_xyz_rad_s[1] * dt_s),
            body.rotation_xyz_rad[2] + (body.angular_velocity_xyz_rad_s[2] * dt_s),
        ]

        body = self.robot.body
        linear_speed_m_s = _v_dot(tuple(body.velocity_xyz_m_s), tuple(body.velocity_xyz_m_s)) ** 0.5
        angular_speed_rad_s = _v_dot(tuple(body.angular_velocity_xyz_rad_s), tuple(body.angular_velocity_xyz_rad_s)) ** 0.5
        if any_grounded and linear_speed_m_s <= self.sleep_linear_speed_threshold_m_s and angular_speed_rad_s <= self.sleep_angular_speed_threshold_rad_s:
            body.velocity_xyz_m_s[:] = [0.0, 0.0, 0.0]
            body.angular_velocity_xyz_rad_s[:] = [0.0, 0.0, 0.0]

        self._update_imu_and_force_state(leg_contacts, body_contacts)

    def _compute_leg_kinematics(self) -> dict[str, dict[str, tuple[float, float, float]]]:
        kinematics: dict[str, dict[str, tuple[float, float, float]]] = {}
        for leg in self.robot.legs:
            mount_world = self._body_point_world(leg.mount_point_xyz_m)
            foot_world = self._body_vector_world(leg.foot_offset_from_mount_m())
            foot_position = _v_add(mount_world, foot_world)
            foot_velocity = _v_add(
                self._point_velocity_world(leg.mount_point_xyz_m),
                self._body_vector_world(leg.foot_velocity_from_mount_m_s()),
            )
            r_world = _v_sub(foot_position, tuple(self.robot.body.position_xyz_m))
            kinematics[leg.name] = {
                "position": foot_position,
                "velocity": foot_velocity,
                "r_world": r_world,
            }
        return kinematics

    def _compute_leg_contacts_from_kinematics(
        self,
        leg_kinematics: dict[str, dict[str, tuple[float, float, float]]],
    ) -> dict[str, dict[str, tuple[float, float, float] | str]]:
        contacts: dict[str, dict[str, tuple[float, float, float] | str]] = {}
        for leg in self.robot.legs:
            info = leg_kinematics[leg.name]
            foot_position = info["position"]
            foot_velocity = info["velocity"]
            force, mode = self._compute_contact_force(
                point_key=leg.name,
                position=foot_position,
                velocity=foot_velocity,
                static_mu=leg.foot_static_friction,
                kinetic_mu=leg.foot_kinetic_friction,
                memory=self._contact_memory[leg.name],
                floor_extra_m=leg.foot_radius_m + self.robot.body.elastic_deformation_m,
            )
            # Contact force acts at the bottom of the spherical foot tip.
            contact_point = (foot_position[0], foot_position[1], foot_position[2] - leg.foot_radius_m)
            r_world = _v_sub(contact_point, tuple(self.robot.body.position_xyz_m))
            contacts[leg.name] = {
                "position": foot_position,
                "velocity": foot_velocity,
                "force": force,
                "r_world": r_world,
                "mode": mode,
            }
        return contacts

    def _compute_body_contacts(self) -> list[dict[str, tuple[float, float, float] | str]]:
        contacts: list[dict[str, tuple[float, float, float] | str]] = []
        body = self.robot.body
        for index, sample_point in enumerate(body.contact_sample_points_body_frame()):
            point_position = self._body_point_world(sample_point)
            point_velocity = self._point_velocity_world(sample_point)
            force, mode = self._compute_contact_force(
                point_key=f"body_{index}",
                position=point_position,
                velocity=point_velocity,
                static_mu=self.body_contact_friction,
                kinetic_mu=self.body_contact_friction,
                memory=self._body_contact_memory[index],
                floor_extra_m=body.elastic_deformation_m,
            )
            if force != (0.0, 0.0, 0.0):
                contacts.append(
                    {
                        "position": point_position,
                        "velocity": point_velocity,
                        "force": force,
                        "r_world": _v_sub(point_position, tuple(body.position_xyz_m)),
                        "mode": mode,
                    }
                )
        return contacts

    def _compute_leg_body_contacts(self) -> list[dict[str, tuple[float, float, float]]]:
        """Contact points along each cylindrical leg body.

        Samples N points along each leg at evenly spaced fractions of the leg length,
        treating each as a sphere of radius leg_radius_m.  When the robot tips onto its
        side the cylindrical leg bodies contact the floor here, preventing the robot from
        staying flat on its side.
        """
        contacts: list[dict[str, tuple[float, float, float]]] = []
        for leg in self.robot.legs:
            foot_offset = leg.foot_offset_from_mount_m()
            foot_vel_body = leg.foot_velocity_from_mount_m_s()
            for i, f in enumerate(_LEG_BODY_FRACTIONS):
                sample_offset_body = _v_scale(foot_offset, f)
                sample_world = _v_add(
                    self._body_point_world(leg.mount_point_xyz_m),
                    self._body_vector_world(sample_offset_body),
                )
                sample_vel_body = _v_scale(foot_vel_body, f)
                sample_vel = _v_add(
                    self._point_velocity_world(leg.mount_point_xyz_m),
                    self._body_vector_world(sample_vel_body),
                )
                memory = self._leg_body_contact_memory[(leg.name, i)]
                force, _ = self._compute_contact_force(
                    point_key=f"{leg.name}_body_{i}",
                    position=sample_world,
                    velocity=sample_vel,
                    static_mu=leg.foot_static_friction,
                    kinetic_mu=leg.foot_kinetic_friction,
                    memory=memory,
                    floor_extra_m=leg.leg_radius_m + self.robot.body.elastic_deformation_m,
                )
                if force != (0.0, 0.0, 0.0):
                    contact_point = (sample_world[0], sample_world[1], sample_world[2] - leg.leg_radius_m)
                    r_world = _v_sub(contact_point, tuple(self.robot.body.position_xyz_m))
                    contacts.append({"force": force, "r_world": r_world})
        return contacts

    def _compute_contact_force(
        self,
        *,
        point_key: str,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        static_mu: float,
        kinetic_mu: float,
        memory: ContactMemory,
        floor_extra_m: float = 0.0,
    ) -> tuple[tuple[float, float, float], str]:
        del point_key
        effective_floor = self.floor_height_m + floor_extra_m
        support_gap = self._rest_contact_buffer_m if memory.is_in_contact else 0.0
        effective_penetration = max((effective_floor + support_gap) - position[2], 0.0)
        normal_force = 0.0
        if effective_penetration > 0.0 or (position[2] <= effective_floor + 1e-5 and velocity[2] <= 0.0):
            unloading_scale = self.unloading_stiffness_scale if velocity[2] > 0.0 else 1.0
            normal_force = (
                (self.normal_stiffness_n_m * unloading_scale * effective_penetration)
                + (self.normal_damping_n_s_m * max(-velocity[2], 0.0))
            )
            normal_force = min(max(normal_force, 0.0), self.max_contact_force_n)

        if normal_force <= 0.0:
            memory.is_in_contact = False
            memory.tangential_anchor_xy_m = (position[0], position[1])
            return ((0.0, 0.0, 0.0), "airborne")

        if not memory.is_in_contact:
            memory.tangential_anchor_xy_m = (position[0], position[1])

        tangential_deflection = (
            position[0] - memory.tangential_anchor_xy_m[0],
            position[1] - memory.tangential_anchor_xy_m[1],
        )
        static_candidate = (
            (-self.tangential_stiffness_n_m * tangential_deflection[0]) - (self.tangential_damping_n_s_m * velocity[0]),
            (-self.tangential_stiffness_n_m * tangential_deflection[1]) - (self.tangential_damping_n_s_m * velocity[1]),
        )
        static_mag = (static_candidate[0] ** 2 + static_candidate[1] ** 2) ** 0.5
        static_limit = static_mu * normal_force

        if static_mag <= static_limit:
            memory.is_in_contact = True
            return ((static_candidate[0], static_candidate[1], normal_force), "static")

        tangential_speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        if tangential_speed > 1e-6:
            friction_xy = (
                -kinetic_mu * normal_force * velocity[0] / tangential_speed,
                -kinetic_mu * normal_force * velocity[1] / tangential_speed,
            )
        else:
            friction_xy = (
                -kinetic_mu * normal_force * _sign(static_candidate[0]),
                -kinetic_mu * normal_force * _sign(static_candidate[1]),
            )

        memory.tangential_anchor_xy_m = (position[0], position[1])
        memory.is_in_contact = True
        return ((friction_xy[0], friction_xy[1], normal_force), "kinetic")

    def _update_imu_and_force_state(
        self,
        leg_contacts: dict[str, dict[str, tuple[float, float, float] | str]],
        body_contacts: list[dict[str, tuple[float, float, float] | str]],
    ) -> None:
        body = self.robot.body
        body.imu.update_rotation(*body.rotation_xyz_rad)
        body.imu.update_acceleration(*body.acceleration_xyz_m_s2)
        self.body_contact_forces_xyz_n = [contact["force"] for contact in body_contacts]

        for leg in self.robot.legs:
            info = leg_contacts.get(leg.name)
            if info is None:
                self.leg_force_states[leg.name] = LegForceState(leg_name=leg.name)
                continue
            force = info["force"]
            acceleration = _v_add(
                tuple(body.acceleration_xyz_m_s2),
                self._body_vector_world(leg.foot_acceleration_from_mount_m_s2()),
            )
            joint_reaction = _v_sub(
                _v_scale(acceleration, leg.mass_kg),
                _v_add(force, (0.0, 0.0, -leg.mass_kg * self.gravity_m_s2)),
            )
            self.leg_force_states[leg.name] = LegForceState(
                leg_name=leg.name,
                contact_mode=str(info["mode"]),
                foot_position_xyz_m=info["position"],
                foot_velocity_xyz_m_s=info["velocity"],
                ground_force_xyz_n=force,
                joint_reaction_xyz_n=joint_reaction,
            )
            leg.imu.update_rotation(body.rotation_xyz_rad[0], leg.angle_rad, body.rotation_xyz_rad[2])
            leg.imu.update_acceleration(*acceleration)

    def _body_rotation_matrix(self) -> tuple[tuple[float, float, float], ...]:
        roll, pitch, yaw = self.robot.body.rotation_xyz_rad
        return _rotation_matrix_xyz(roll, pitch, yaw)

    def _body_vector_world(self, vector_body: tuple[float, float, float]) -> tuple[float, float, float]:
        return _mat_vec_mul(self._body_rotation_matrix(), vector_body)

    def _body_point_world(self, point_body: tuple[float, float, float]) -> tuple[float, float, float]:
        return _v_add(tuple(self.robot.body.position_xyz_m), self._body_vector_world(point_body))

    def _point_velocity_world(self, point_body: tuple[float, float, float]) -> tuple[float, float, float]:
        body = self.robot.body
        r_world = self._body_vector_world(point_body)
        omega = tuple(body.angular_velocity_xyz_rad_s)
        return _v_add(tuple(body.velocity_xyz_m_s), _v_cross(omega, r_world))

    def center_of_mass_xyz_m(self) -> tuple[float, float, float]:
        body = self.robot.body
        total_mass = self.robot.total_mass_kg
        weighted = _v_scale(tuple(body.position_xyz_m), body.mass_kg)
        for leg in self.robot.legs:
            mount_world = self._body_point_world(leg.mount_point_xyz_m)
            leg_com_world = _v_add(mount_world, self._body_vector_world(leg.com_offset_from_mount_m()))
            weighted = _v_add(weighted, _v_scale(leg_com_world, leg.mass_kg))
        return _v_scale(weighted, 1.0 / total_mass)
