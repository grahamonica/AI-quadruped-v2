"""Polymorphic playback for the live viewer.

The live viewer drives MuJoCo from either a trained policy or a passive reset
pose. Different policies (shared-trunk ES, discrete-leg linear) have different
parameter shapes, action layers, and brain dts. This module exposes a single
``Playback`` interface so the viewer thread doesn't have to know which kind of
model it's running.

A Playback instance gives the viewer:
- ``spec``: the runtime spec (used for metadata broadcast and dt pacing).
- ``rollout_backend``: the MuJoCoBackend used to set up the viewer renderer.
- ``state``: a small mutable bag matching the fields the viewer broadcasts.
- ``top_rewards``: a numpy array of recent best returns (may be empty).
- ``random_goal()``: produce a goal XYZ for a fresh episode.
- ``advance_key()``: hand the caller a fresh JAX PRNGKey.
- ``run_logged_episode(goal_xyz, key, on_step, steps)``: execute one episode,
  invoking ``on_step(step_message)`` per brain step.

The factory ``load_playback_for_selection`` dispatches on the artifact's
manifest (or, in legacy cases, on the npz contents) to return the right
implementation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from brains.config import RuntimeSpec, load_runtime_spec
from brains.runtime.model_store import (
    MANIFEST_FILENAME,
    ModelArtifact,
    find_model_artifact,
    runtime_spec_from_checkpoint,
)


StepCallback = Callable[[dict[str, Any]], None]


@dataclass
class PlaybackState:
    """Mutable rollout state the viewer broadcasts in the ``generation`` message."""

    generation: int = 0
    mean_reward: float = 0.0
    best_reward: float = 0.0
    rewards_history: list[float] = field(default_factory=list)
    goal_xyz: tuple[float, float, float] = (0.0, 0.0, 0.16)


class Playback:
    """Common interface for all playback paths.

    Subclasses must populate ``spec``, ``rollout_backend``, ``state``, and
    ``top_rewards`` and implement ``random_goal``, ``advance_key``, and
    ``run_logged_episode``.
    """

    spec: RuntimeSpec
    rollout_backend: Any
    state: PlaybackState
    top_rewards: np.ndarray
    architecture: str = "unknown"

    def random_goal(self) -> jax.Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def advance_key(self) -> jax.Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_logged_episode(
        self,
        goal_xyz: jax.Array,
        key: jax.Array,
        on_step: StepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class EsTrainerPlayback(Playback):
    """Wraps the existing ``ESTrainer`` for shared-trunk and command-primitive policies."""

    architecture = "shared_trunk_es"

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer
        self.spec = trainer.spec
        self.rollout_backend = trainer._rollout_backend
        self.state = PlaybackState(
            generation=int(getattr(trainer.state, "generation", 0)),
            mean_reward=float(getattr(trainer.state, "mean_reward", 0.0)),
            best_reward=float(getattr(trainer.state, "best_reward", 0.0)),
            rewards_history=list(getattr(trainer.state, "rewards_history", [])),
            goal_xyz=tuple(float(x) for x in getattr(trainer.state, "goal_xyz", (0.0, 0.0, 0.16))),
        )
        self.top_rewards = np.asarray(getattr(trainer, "top_rewards", np.array([])), dtype=np.float32)

    def random_goal(self) -> jax.Array:
        return self._trainer._random_goal()

    def advance_key(self) -> jax.Array:
        self._trainer._key, episode_key = jax.random.split(self._trainer._key)
        return episode_key

    def run_logged_episode(
        self,
        goal_xyz: jax.Array,
        key: jax.Array,
        on_step: StepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:
        params = np.asarray(self._trainer.params, dtype=np.float32)
        result = self.rollout_backend.run_logged_episode(
            params,
            goal_xyz,
            key,
            on_step,
            steps=int(steps),
            spawn_xy=spawn_xy,
        )
        # Sync state back from trainer in case the rollout updated anything visible.
        self.state.generation = int(getattr(self._trainer.state, "generation", self.state.generation))
        self.state.mean_reward = float(getattr(self._trainer.state, "mean_reward", self.state.mean_reward))
        self.state.best_reward = float(getattr(self._trainer.state, "best_reward", self.state.best_reward))
        self.state.rewards_history = list(getattr(self._trainer.state, "rewards_history", self.state.rewards_history))
        self.state.goal_xyz = tuple(float(x) for x in getattr(self._trainer.state, "goal_xyz", self.state.goal_xyz))
        self.top_rewards = np.asarray(getattr(self._trainer, "top_rewards", self.top_rewards), dtype=np.float32)
        return float(result)


class StillPlayback(Playback):
    """Renders the MuJoCo reset pose without applying a policy or controls."""

    architecture = "static_scene"
    static = True

    def __init__(self, spec: RuntimeSpec) -> None:
        from brains.sim.mujoco_backend import MuJoCoBackend

        self.spec = spec
        self.rollout_backend = MuJoCoBackend(spec)
        self.state = PlaybackState(
            goal_xyz=(float(spec.goals.radius_m), 0.0, float(spec.goals.height_m)),
        )
        self.top_rewards = np.zeros((0,), dtype=np.float32)

    def random_goal(self) -> jax.Array:
        if self.spec.goals.strategy == "fixed" and self.spec.goals.fixed_goal_xyz is not None:
            return jnp.asarray(self.spec.goals.fixed_goal_xyz, dtype=jnp.float32)
        return jnp.asarray(self.state.goal_xyz, dtype=jnp.float32)

    def advance_key(self) -> jax.Array:
        return jax.random.PRNGKey(0)

    def run_logged_episode(
        self,
        goal_xyz: jax.Array,
        key: jax.Array,
        on_step: StepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:
        del key, steps
        goal_np = np.asarray(goal_xyz, dtype=np.float32)
        if goal_np.shape[-1] == 2:
            goal_np = np.asarray([goal_np[0], goal_np[1], float(self.spec.goals.height_m)], dtype=np.float32)
        spawn_np = np.zeros((2,), dtype=np.float32) if spawn_xy is None else np.asarray(spawn_xy, dtype=np.float32)
        data = self.rollout_backend.reset_data(spawn_xy=spawn_np)
        metrics = self.rollout_backend.initial_metrics(data, goal_np)
        snapshot = self.rollout_backend._snapshot(data, metrics, goal_np, step_index=0, total_steps=1)
        snapshot["action_mode"] = "still"
        snapshot["selected_command"] = None
        snapshot["reward"] = 0.0
        on_step(snapshot)
        self.state.goal_xyz = tuple(float(value) for value in goal_np.tolist())
        return 0.0


class DiscreteLegPlayback(Playback):
    """Plays back a discrete-leg linear policy through ``DiscreteLegHarness``.

    The checkpoint stores ``W (num_actions, encoded_obs_dim)``, ``b
    (num_actions,)`` and the harness configuration (brain_dt, encoding gain,
    goal radius). On each brain step we encode the harness observation,
    argmax the linear policy, advance the harness, and emit a snapshot in
    the same shape ``MuJoCoBackend._snapshot`` produces so the viewer's
    per-frame renderer can drop in unchanged.
    """

    architecture = "discrete_leg_linear"

    def __init__(
        self,
        spec: RuntimeSpec,
        params: dict[str, np.ndarray],
        harness_config: dict[str, Any],
        seed: int,
        artifact_metrics: dict[str, Any] | None = None,
    ) -> None:
        from brains.harnesses import DiscreteLegHarness

        self.spec = spec
        # Float32 round-tripping (e.g. brain_dt_s=0.20 saved as float32 then read back) drifts
        # by ~1e-7 which fails MuJoCoBackend's strict integer-substep check. Snap brain_dt_s to
        # the nearest exact multiple of the simulator timestep before constructing the harness.
        timestep_s = float(spec.simulator.mujoco.timestep_s)
        raw_brain_dt = float(harness_config.get("brain_dt_s", 0.20))
        substeps = max(1, round(raw_brain_dt / timestep_s))
        snapped_brain_dt = float(substeps) * timestep_s
        self._harness = DiscreteLegHarness(
            spec=spec,
            brain_dt_s=snapped_brain_dt,
            goal_reached_radius_m=float(harness_config.get("goal_reached_radius_m", 0.30)),
            tip_kill_depth=float(harness_config.get("tip_kill_depth", 0.7)),
            tip_kill_steps=int(harness_config.get("tip_kill_steps", 2)),
            positional_encoding_gain=float(harness_config.get("positional_encoding_gain", 0.35)),
            goal_reached_bonus=float(harness_config.get("goal_reached_bonus", 5.0)),
            tipped_penalty=float(harness_config.get("tipped_penalty", 0.0)),
        )
        # Force the harness to build its backend now so renderer setup can use it.
        self._harness.reset(spawn_xy=np.zeros((2,), dtype=np.float32))
        self.rollout_backend = self._harness._backend
        self.spec = self._harness.spec  # may include the brain_dt override
        self._params_W = jnp.asarray(params["W"], dtype=jnp.float32)
        self._params_b = jnp.asarray(params["b"], dtype=jnp.float32)
        metrics = artifact_metrics or {}
        self.state = PlaybackState(
            generation=int(metrics.get("generation", 0)),
            mean_reward=float(metrics.get("mean_reward") or 0.0),
            best_reward=float(metrics.get("best_reward") or 0.0),
            rewards_history=[],
            goal_xyz=(float(spec.goals.radius_m), 0.0, float(spec.goals.height_m)),
        )
        self.top_rewards = np.zeros((0,), dtype=np.float32)
        self._key = jax.random.PRNGKey(int(seed))

    def random_goal(self) -> jax.Array:
        if self.spec.goals.strategy == "fixed" and self.spec.goals.fixed_goal_xyz is not None:
            return jnp.asarray(self.spec.goals.fixed_goal_xyz, dtype=jnp.float32)
        self._key, angle_key = jax.random.split(self._key)
        angle = jax.random.uniform(angle_key, (), minval=0.0, maxval=2.0 * math.pi)
        radius = float(self.spec.goals.radius_m)
        return jnp.asarray(
            [
                radius * jnp.cos(angle),
                radius * jnp.sin(angle),
                float(self.spec.goals.height_m),
            ],
            dtype=jnp.float32,
        )

    def advance_key(self) -> jax.Array:
        self._key, episode_key = jax.random.split(self._key)
        return episode_key

    def _logits(self, encoded_obs: jax.Array) -> jax.Array:
        return self._params_W @ encoded_obs + self._params_b

    def run_logged_episode(
        self,
        goal_xyz: jax.Array,
        key: jax.Array,
        on_step: StepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:
        del key  # discrete policy is deterministic given state; harness ignores this key
        goal_np = np.asarray(goal_xyz, dtype=np.float32)
        if goal_np.shape[-1] == 2:
            goal_np = np.asarray([goal_np[0], goal_np[1], float(self.spec.goals.height_m)], dtype=np.float32)
        spawn_np = np.zeros((2,), dtype=np.float32) if spawn_xy is None else np.asarray(spawn_xy, dtype=np.float32)
        obs = self._harness.reset(spawn_xy=spawn_np, goal_xyz=goal_np)
        backend = self._harness._backend
        data = self._harness._data
        metrics = backend.initial_metrics(data, goal_np)
        total_reward = 0.0
        for step_index in range(int(steps)):
            encoded = self._harness.encode_obs(obs)
            logits = self._logits(encoded)
            action_index = int(jnp.argmax(logits))
            result = self._harness.step(action_index)
            total_reward += float(result.reward)
            metrics = backend._step_metrics(data, metrics, goal_np)
            snapshot = backend._snapshot(data, metrics, goal_np, step_index, int(steps))
            snapshot["action_mode"] = "discrete_leg"
            deltas = ",".join(str(int(d)) for d in result.info.get("deltas", []))
            snapshot["selected_command"] = f"a{action_index} ({deltas})"
            snapshot["reward"] = float(total_reward)
            on_step(snapshot)
            if result.done:
                break
            obs = result.obs
        return float(total_reward)


def _read_manifest(artifact: ModelArtifact) -> dict[str, Any]:
    if artifact.manifest_path is not None and artifact.manifest_path.exists():
        try:
            return json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _checkpoint_architecture(npz_path: Path) -> str | None:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = set(data.files)
    except Exception:
        return None
    if {"W", "b", "num_actions"}.issubset(keys):
        return "discrete_leg_linear"
    if {"W", "b"}.issubset(keys) and "num_actions" in keys:
        return "discrete_leg_linear"
    return None


def _detect_architecture(artifact: ModelArtifact) -> str:
    manifest = _read_manifest(artifact)
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    model = runtime.get("model", {}) if isinstance(runtime, dict) else {}
    architecture = model.get("architecture")
    if isinstance(architecture, str) and architecture:
        return architecture
    detected = _checkpoint_architecture(artifact.checkpoint_path)
    if detected is not None:
        return detected
    return "shared_trunk_es"


def _load_discrete_leg_playback(
    artifact: ModelArtifact,
    config_path: Path,
    seed: int,
) -> DiscreteLegPlayback:
    with np.load(artifact.checkpoint_path, allow_pickle=False) as data:
        params = {"W": np.asarray(data["W"]), "b": np.asarray(data["b"])}
        harness_config = {
            key: float(data[key].item()) if data[key].shape == () else float(np.asarray(data[key]).item())
            for key in ("brain_dt_s", "positional_encoding_gain", "goal_reached_radius_m")
            if key in data.files
        }
    base_spec = runtime_spec_from_checkpoint(artifact.checkpoint_path) or load_runtime_spec(config_path)
    metrics = {
        "generation": int(artifact.generation or 0),
        "best_reward": artifact.best_reward,
        "mean_reward": artifact.mean_reward,
    }
    return DiscreteLegPlayback(
        spec=base_spec,
        params=params,
        harness_config=harness_config,
        seed=seed,
        artifact_metrics=metrics,
    )


def _load_es_trainer_playback(
    artifact: ModelArtifact | None,
    config_path: Path,
    seed: int,
) -> tuple[EsTrainerPlayback, Path | None]:
    from brains.jax_trainer import ESTrainer, apply_runtime_spec

    checkpoint_path = artifact.checkpoint_path if artifact is not None else None
    spec = runtime_spec_from_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    if spec is None:
        spec = load_runtime_spec(config_path)
    apply_runtime_spec(spec)
    trainer = ESTrainer(
        seed=seed,
        spec=spec,
        model_id=artifact.id if artifact is not None else None,
        log_id=artifact.log_id if artifact is not None else None,
    )
    loaded_checkpoint: Path | None = None
    if checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path)
        trainer.model_id = artifact.id if artifact is not None else trainer.model_id
        trainer.log_id = artifact.log_id if artifact is not None else trainer.log_id
        loaded_checkpoint = checkpoint_path
    return EsTrainerPlayback(trainer), loaded_checkpoint


@dataclass
class PlaybackLoadResult:
    playback: Playback
    artifact: ModelArtifact | None
    loaded_checkpoint: Path | None
    skipped: tuple[dict[str, str], ...]


def load_playback_for_selection(
    config_path: Path,
    seed: int,
    selected_model_id: str | None,
    *,
    checkpoint_root: str | Path = "checkpoints",
) -> PlaybackLoadResult:
    """Load the right Playback for the given model id.

    Dispatches on the artifact's manifest (``runtime.model.architecture``) and
    falls back to inspecting the npz file. Anything that doesn't match a
    known specialty defaults to the shared-trunk ES path. No selection returns
    a passive scene playback instead of an untrained policy.
    """

    if not selected_model_id:
        return PlaybackLoadResult(
            playback=StillPlayback(load_runtime_spec(config_path)),
            artifact=None,
            loaded_checkpoint=None,
            skipped=(),
        )

    artifact = (
        find_model_artifact(selected_model_id, checkpoint_root)
        if selected_model_id
        else None
    )
    skipped: list[dict[str, str]] = []
    if artifact is None:
        return PlaybackLoadResult(
            playback=StillPlayback(load_runtime_spec(config_path)),
            artifact=None,
            loaded_checkpoint=None,
            skipped=({"path": str(selected_model_id), "reason": "model artifact not found"},),
        )

    if artifact is not None:
        try:
            architecture = _detect_architecture(artifact)
        except Exception as exc:
            skipped.append({"path": str(artifact.checkpoint_path), "reason": f"architecture probe failed: {exc}"})
            architecture = "shared_trunk_es"

        if architecture == "discrete_leg_linear":
            try:
                playback = _load_discrete_leg_playback(artifact, config_path, seed)
                return PlaybackLoadResult(
                    playback=playback,
                    artifact=artifact,
                    loaded_checkpoint=artifact.checkpoint_path,
                    skipped=tuple(skipped),
                )
            except Exception as exc:
                skipped.append({"path": str(artifact.checkpoint_path), "reason": str(exc)})
                # fall through to the ES trainer path so the viewer still has something to render

    try:
        playback, loaded_checkpoint = _load_es_trainer_playback(artifact, config_path, seed)
    except Exception as exc:
        if artifact is not None:
            skipped.append({"path": str(artifact.checkpoint_path), "reason": str(exc)})
        playback = StillPlayback(load_runtime_spec(config_path))
        loaded_checkpoint = None

    return PlaybackLoadResult(
        playback=playback,
        artifact=artifact,
        loaded_checkpoint=loaded_checkpoint,
        skipped=tuple(skipped),
    )
