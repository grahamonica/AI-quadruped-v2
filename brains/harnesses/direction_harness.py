"""Scripted direction harness for quadruped control experiments.

This module is intentionally not connected to trainable models. It gives us a
small command vocabulary and a MuJoCo playback path for testing higher-level
interfaces before those interfaces become model inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from brains.config import DEFAULT_CONFIG_PATH, RuntimeSpec, load_runtime_spec
from brains.sim.action_layer import command_target_velocity


StepCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class HarnessOption:
    name: str
    aliases: tuple[str, ...]
    description: str
    default_duration_s: float
    default_speed: float


@dataclass(frozen=True)
class MotionCommand:
    option: str
    duration_s: float
    speed: float = 1.0


@dataclass(frozen=True)
class CommandPlan:
    commands: tuple[MotionCommand, ...]

    @property
    def duration_s(self) -> float:
        return sum(command.duration_s for command in self.commands)

    def command_at(self, time_s: float) -> MotionCommand:
        if not self.commands:
            raise ValueError("command plan has no commands.")
        elapsed = 0.0
        for command in self.commands:
            elapsed += command.duration_s
            if time_s <= elapsed:
                return command
        return self.commands[-1]


@dataclass(frozen=True)
class HarnessRun:
    plan: CommandPlan
    frames: tuple[dict[str, Any], ...]
    total_reward: float


class DirectionHarness:
    """Translate plain directions into simple scripted motor targets."""

    OPTIONS: tuple[HarnessOption, ...] = (
        HarnessOption("stand", ("stand", "hold", "idle", "stay"), "Hold all leg velocity targets at zero.", 1.0, 0.0),
        HarnessOption("trot", ("trot", "forward", "go forward"), "Alternating diagonal gait.", 2.0, 0.50),
        HarnessOption("walk", ("walk", "slow walk", "stroll"), "Slower four-phase stepping pattern.", 2.0, 0.40),
        HarnessOption("skip", ("skip", "hop"), "Faster asymmetric skipping cadence.", 1.6, 0.45),
        HarnessOption("turn_left", ("turn left", "left", "rotate left"), "Stabilized yaw-left gait.", 1.5, 0.55),
        HarnessOption("turn_right", ("turn right", "right", "rotate right"), "Stabilized yaw-right gait.", 1.5, 0.55),
        HarnessOption("front_flip", ("front flip", "forward flip", "frontflip"), "Aggressive front-flip impulse pattern.", 1.0, 1.0),
        HarnessOption("back_flip", ("back flip", "backward flip", "backflip"), "Aggressive back-flip impulse pattern.", 1.0, 1.0),
        HarnessOption("side_roll", ("side roll", "roll", "barrel roll"), "Lateral rolling impulse pattern.", 1.0, 1.0),
        HarnessOption("back_up", ("back up", "back", "reverse"), "Run the trot phase in reverse.", 1.5, 0.4),
        HarnessOption("stop", ("stop", "halt", "pause"), "End with zero motor targets.", 0.75, 0.0),
    )

    def __init__(self, spec: RuntimeSpec | str | Path | None = None) -> None:
        if spec is None:
            self.spec = load_runtime_spec(DEFAULT_CONFIG_PATH)
        elif isinstance(spec, RuntimeSpec):
            self.spec = spec
        else:
            self.spec = load_runtime_spec(Path(spec))
        self._options_by_name = {option.name: option for option in self.OPTIONS}
        self._aliases = {
            alias: option.name
            for option in self.OPTIONS
            for alias in (option.name, *option.aliases)
        }

    def available_options(self) -> tuple[HarnessOption, ...]:
        return self.OPTIONS

    def compile(self, direction: str, *, duration_s: float | None = None, speed: float | None = None) -> CommandPlan:
        text = " ".join(direction.lower().replace(",", " ").split())
        if not text:
            return CommandPlan((MotionCommand("stand", duration_s or 1.0, 0.0),))

        selected: list[MotionCommand] = []
        for alias in sorted(self._aliases, key=len, reverse=True):
            if alias in text:
                option_name = self._aliases[alias]
                option = self._options_by_name[option_name]
                selected.append(
                    MotionCommand(
                        option=option_name,
                        duration_s=float(duration_s or option.default_duration_s),
                        speed=float(speed if speed is not None else option.default_speed),
                    )
                )

        if not selected:
            known = ", ".join(option.name for option in self.OPTIONS)
            raise ValueError(f"Could not map direction {direction!r} to a harness option. Known options: {known}.")

        deduped: list[MotionCommand] = []
        seen: set[str] = set()
        for command in selected:
            if command.option in seen:
                continue
            seen.add(command.option)
            deduped.append(command)
        return CommandPlan(tuple(deduped))

    def target_velocity(self, option_name: str, time_s: float, *, speed: float | None = None) -> np.ndarray:
        if option_name not in self._options_by_name:
            raise ValueError(f"Unknown harness option {option_name!r}.")
        option = self._options_by_name[option_name]
        resolved_speed = float(option.default_speed if speed is None else speed)
        command_name = "stand" if option_name == "stop" else option_name
        return command_target_velocity(
            command_name,
            time_s=time_s,
            speed=resolved_speed,
            max_motor_rad_s=float(self.spec.robot.max_motor_rad_s),
        )

    def default_goal(self) -> np.ndarray:
        return np.asarray([self.spec.goals.radius_m, 0.0, self.spec.goals.height_m], dtype=np.float32)

    def _make_backend(self):
        from brains.sim.mujoco_backend import MuJoCoBackend

        return MuJoCoBackend(self.spec)

    def run(
        self,
        direction: str | CommandPlan,
        *,
        steps: int | None = None,
        goal_xyz: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        on_step: StepCallback | None = None,
    ) -> HarnessRun:
        plan = direction if isinstance(direction, CommandPlan) else self.compile(direction)
        backend = self._make_backend()
        data = backend.reset_data(spawn_xy=spawn_xy)
        goal = self.default_goal() if goal_xyz is None else np.asarray(goal_xyz, dtype=np.float32)
        metrics = backend.initial_metrics(data, goal)
        total_steps = int(steps if steps is not None else max(1, math.ceil(plan.duration_s / self.spec.episode.brain_dt_s)))
        frames: list[dict[str, Any]] = []

        for step_index in range(total_steps):
            time_s = step_index * float(self.spec.episode.brain_dt_s)
            command = plan.command_at(time_s)
            target_velocity = self.target_velocity(command.option, time_s, speed=command.speed)
            backend._advance(data, target_velocity)
            metrics = backend._step_metrics(data, metrics, goal)
            frame = backend._snapshot(data, metrics, goal, step_index, total_steps)
            frame["harness"] = {
                "option": command.option,
                "target_velocity_rad_s": target_velocity.tolist(),
            }
            frames.append(frame)
            if on_step is not None:
                on_step(frame)

        return HarnessRun(plan=plan, frames=tuple(frames), total_reward=float(metrics.total_reward))
