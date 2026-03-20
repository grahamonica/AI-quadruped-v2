from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from .environment import QuadrupedEnvironment
from .robot import Quadruped


class QuadrupedDemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("KT2-Style Quadruped 3D Physics Demo")

        self.robot = Quadruped.create_kt2_style()
        self.environment = QuadrupedEnvironment(self.robot)
        self.last_frame_time_s = time.perf_counter()
        self.motor_controls: dict[str, tk.DoubleVar] = {}

        self._build_ui()
        self._draw_scene()
        self._update_telemetry()
        self._schedule_frame()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(column=0, row=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(container, width=1000, height=620, background="#f3efe4", highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky="nsew", padx=(0, 12))

        controls = ttk.Frame(container)
        controls.grid(column=1, row=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(10, weight=1)

        ttk.Label(controls, text="Motor Velocity Control (rad/s)", font=("Courier", 12, "bold")).grid(column=0, row=0, sticky="w", pady=(0, 8))

        for index, leg in enumerate(self.robot.legs, start=1):
            value = tk.DoubleVar(value=0.0)
            self.motor_controls[leg.name] = value
            ttk.Label(controls, text=leg.name, font=("Courier", 10)).grid(column=0, row=(index * 2) - 1, sticky="w")
            tk.Scale(
                controls,
                from_=-8.0,
                to=8.0,
                resolution=0.1,
                orient="horizontal",
                variable=value,
                length=280,
                command=lambda _v, leg_name=leg.name: self._update_motor_target(leg_name),
            ).grid(column=0, row=index * 2, sticky="ew", pady=(0, 6))

        button_row = ttk.Frame(controls)
        button_row.grid(column=0, row=9, sticky="ew", pady=(8, 10))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        ttk.Button(button_row, text="Zero Motors", command=self._zero_motors).grid(column=0, row=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_row, text="Reset Pose", command=self._reset_pose).grid(column=1, row=0, sticky="ew", padx=(4, 0))

        self.telemetry = tk.Text(controls, width=48, height=22, wrap="none", font=("Courier", 10), background="#fffdf8")
        self.telemetry.grid(column=0, row=10, sticky="nsew")

    def _update_motor_target(self, leg_name: str) -> None:
        self.robot.set_leg_motor_velocity(leg_name, self.motor_controls[leg_name].get())

    def _zero_motors(self) -> None:
        for leg_name, control in self.motor_controls.items():
            control.set(0.0)
            self.robot.set_leg_motor_velocity(leg_name, 0.0)

    def _reset_pose(self) -> None:
        self.environment.reset_pose()
        self._zero_motors()

    def _schedule_frame(self) -> None:
        self._frame()
        self.root.after(16, self._schedule_frame)

    def _frame(self) -> None:
        now_s = time.perf_counter()
        dt_s = min(now_s - self.last_frame_time_s, 1.0 / 30.0)
        self.last_frame_time_s = now_s
        for leg_name, control in self.motor_controls.items():
            self.robot.set_leg_motor_velocity(leg_name, control.get())
        self.environment.advance(dt_s)
        self._draw_scene()
        self._update_telemetry()

    def _project(self, point: tuple[float, float, float]) -> tuple[float, float]:
        x, y, z = point
        screen_x = 500.0 + ((x - y) * 260.0)
        screen_y = 520.0 - (z * 520.0) + ((x + y) * 90.0)
        return (screen_x, screen_y)

    def _draw_scene(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1000, 620, fill="#f3efe4", outline="")

        floor = [(-0.9, -0.6, 0.0), (0.9, -0.6, 0.0), (0.9, 0.6, 0.0), (-0.9, 0.6, 0.0)]
        floor_points = [coord for p in floor for coord in self._project(p)]
        self.canvas.create_polygon(*floor_points, fill="#d6ccb7", outline="#8c7d67", width=2)

        body = self.robot.body
        corners = [self.environment._body_point_world(corner) for corner in body.corners_body_frame()]
        edge_indices = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
            (4, 5), (4, 6), (5, 7), (6, 7),
        ]
        for i0, i1 in edge_indices:
            p0 = self._project(corners[i0])
            p1 = self._project(corners[i1])
            self.canvas.create_line(*p0, *p1, fill="#4d3724", width=3)

        for leg in self.robot.legs:
            state = self.environment.leg_force_states[leg.name]
            mount = self.environment._body_point_world(leg.mount_point_xyz_m)
            foot = state.foot_position_xyz_m
            color = "#666666"
            if state.contact_mode == "static":
                color = "#1f7a4d"
            elif state.contact_mode == "kinetic":
                color = "#c86b1f"
            # Draw cylindrical leg body with hemispherical foot tip.
            # capstyle=ROUND draws a semicircular cap at the foot end — that IS the hemisphere.
            # Cap radius = width/2 = 4px, matching the cylinder cross-section.
            self.canvas.create_line(
                *self._project(mount), *self._project(foot),
                fill=color, width=8, capstyle=tk.ROUND,
            )

        com = self.environment.center_of_mass_xyz_m()
        com_xy = self._project(com)
        self.canvas.create_oval(com_xy[0] - 5, com_xy[1] - 5, com_xy[0] + 5, com_xy[1] + 5, fill="#111111", outline="")
        self.canvas.create_text(com_xy[0] + 10, com_xy[1] - 10, text="COM", anchor="w", font=("Courier", 9))

        support_points = [
            self.environment.leg_force_states[leg.name].foot_position_xyz_m
            for leg in self.robot.legs
            if self.environment.leg_force_states[leg.name].ground_force_xyz_n[2] > 0.0
        ]
        for point in support_points:
            px, py = self._project((point[0], point[1], 0.0))
            self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#2c6fb7", outline="")

    def _update_telemetry(self) -> None:
        body = self.robot.body
        com = self.environment.center_of_mass_xyz_m()
        lines = [
            f"time_s               {self.environment.time_s:8.3f}",
            f"body_pos_xyz_m       ({body.position_xyz_m[0]:6.3f}, {body.position_xyz_m[1]:6.3f}, {body.position_xyz_m[2]:6.3f})",
            f"body_vel_xyz_m_s     ({body.velocity_xyz_m_s[0]:6.3f}, {body.velocity_xyz_m_s[1]:6.3f}, {body.velocity_xyz_m_s[2]:6.3f})",
            f"body_rot_rpy_rad     ({body.rotation_xyz_rad[0]:6.3f}, {body.rotation_xyz_rad[1]:6.3f}, {body.rotation_xyz_rad[2]:6.3f})",
            f"body_omega_xyz       ({body.angular_velocity_xyz_rad_s[0]:6.3f}, {body.angular_velocity_xyz_rad_s[1]:6.3f}, {body.angular_velocity_xyz_rad_s[2]:6.3f})",
            f"com_xyz_m            ({com[0]:6.3f}, {com[1]:6.3f}, {com[2]:6.3f})",
            "",
        ]
        for leg in self.robot.legs:
            state = self.environment.leg_force_states[leg.name]
            lines.extend(
                [
                    f"{leg.name}",
                    f"  target_vel_rad_s  {leg.motor.target_velocity_rad_s:8.3f}",
                    f"  angle_rad         {leg.angle_rad:8.3f}",
                    f"  mode              {state.contact_mode}",
                    f"  foot_xyz_m        ({state.foot_position_xyz_m[0]:6.3f}, {state.foot_position_xyz_m[1]:6.3f}, {state.foot_position_xyz_m[2]:6.3f})",
                    f"  ground_force_n    ({state.ground_force_xyz_n[0]:6.2f}, {state.ground_force_xyz_n[1]:6.2f}, {state.ground_force_xyz_n[2]:6.2f})",
                ]
            )
        self.telemetry.configure(state="normal")
        self.telemetry.delete("1.0", tk.END)
        self.telemetry.insert("1.0", "\n".join(lines))
        self.telemetry.configure(state="disabled")


def run_demo() -> None:
    root = tk.Tk()
    QuadrupedDemoApp(root)
    root.mainloop()
