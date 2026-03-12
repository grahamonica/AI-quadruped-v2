"""Headless training entry point.

Runs ES training without starting the websocket server or React frontend and
saves periodic checkpoints to disk.
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np

import ai.jax_trainer as trainer_module
from ai.trainer import ESTrainer

GPU_DEFAULT_POP_SIZE = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless quadruped training and save checkpoints.")
    parser.add_argument("--generations", type=int, default=10000, help="Number of generations to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trainer initialization.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for checkpoint files.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Retained for CLI compatibility; numbered generation checkpoints are no longer written.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing .npz checkpoint to resume from.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=None,
        help="Override simulated seconds per episode for faster headless experiments.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override ES population size.",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=0,
        help="If > 0, print headless progress every N streamed episode steps.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start_time_s = time.perf_counter()

    def _log(message: str) -> None:
        elapsed_s = time.perf_counter() - start_time_s
        print(f"[elapsed={elapsed_s:9.2f}s] {message}", flush=True)

    if args.episode_seconds is not None:
        trainer_module.EPISODE_S = args.episode_seconds
    if args.population_size is not None:
        trainer_module.POP_SIZE = args.population_size

    trainer = ESTrainer(seed=args.seed)
    if args.population_size is None and getattr(trainer, "backend", "unknown") == "gpu" and trainer_module.POP_SIZE < GPU_DEFAULT_POP_SIZE:
        trainer_module.POP_SIZE = GPU_DEFAULT_POP_SIZE

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        _log(f"Resumed from {args.resume}")

    latest_path = args.out_dir / "latest.npz"
    best_path = args.out_dir / "best.npz"
    top_paths = [args.out_dir / f"top_{rank:02d}.npz" for rank in range(1, trainer_module.PARENT_ELITE_COUNT + 1)]
    best_single_path = args.out_dir / "best_single.npz"
    interrupted = False

    def _save_checkpoint(path: Path, label: str) -> Path:
        saved_path = trainer.save_checkpoint(path)
        _log(
            f"checkpoint updated: {label:<10} "
            f"gen={trainer.state.generation:6d}  "
            f"path={saved_path}"
        )
        return saved_path

    def _save_payload(path: Path, label: str, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        _log(
            f"checkpoint updated: {label:<10} "
            f"gen={trainer.state.generation:6d}  "
            f"path={path}"
        )
        return path

    def _candidate_payload(rank_index: int) -> dict:
        payload = trainer.checkpoint_dict()
        top_params = trainer.top_params
        top_rewards = trainer.top_rewards
        top_indices = trainer.top_indices
        top_generations = trainer.top_generations
        payload["params"] = top_params[rank_index].astype(np.float32)
        payload["candidate_reward"] = np.float32(top_rewards[rank_index])
        payload["candidate_rank"] = np.int32(rank_index + 1)
        payload["candidate_source_index"] = np.int32(top_indices[rank_index])
        payload["candidate_source_generation"] = np.int32(top_generations[rank_index])
        return payload

    def _save_top_ranked() -> None:
        top_rewards = trainer.top_rewards
        for rank_index, path in enumerate(top_paths):
            if rank_index < top_rewards.shape[0]:
                _save_payload(path, f"top_{rank_index + 1:02d}", _candidate_payload(rank_index))
            else:
                path.unlink(missing_ok=True)

    def _handle_interrupt(_sig, _frame) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    _log("Headless training started")
    backend_description = f"JAX ES trainer ({getattr(trainer, 'backend', 'unknown')})"
    _log(f"Backend: {backend_description}. No frontend or rendering is started.")
    if hasattr(trainer, "device_summary"):
        _log(f"Devices: {trainer.device_summary}")
    if getattr(trainer, "backend", "unknown") != "gpu":
        _log("Warning: JAX is not using a GPU. This run will not use Nvidia acceleration.")
    elif args.population_size is None:
        _log(f"GPU detected. Using larger default population size {trainer_module.POP_SIZE} for better device utilization.")
    if args.progress_every_steps > 0:
        _log("Warning: step progress logging is enabled and will reduce training throughput substantially.")
    _log(
        f"Config: episode_s={trainer_module.EPISODE_S}  pop_size={trainer_module.POP_SIZE}  "
        f"checkpoints=latest,best,best_single,top_01-top_0{trainer_module.PARENT_ELITE_COUNT}"
    )
    previous_best = trainer.state.best_reward
    previous_best_single = trainer.state.best_single_reward

    for _ in range(args.generations):
        if interrupted:
            break

        generation_start_s = time.perf_counter()
        target_generation = trainer.state.generation + 1

        def _on_step(msg: dict) -> None:
            if args.progress_every_steps <= 0:
                return
            step = int(msg.get("step", 0))
            total_steps = int(msg.get("total_steps", 0))
            if step == 0 or step % args.progress_every_steps == 0 or step >= max(total_steps - 1, 0):
                _log(
                    f"gen={target_generation:6d}  "
                    f"step={step:6d}/{total_steps:6d}  "
                    f"reward={float(msg.get('reward', 0.0)):10.3f}  "
                    f"sim={float(msg.get('time_s', 0.0)):8.2f}s"
                )

        trainer.run_generation(on_step=_on_step if args.progress_every_steps > 0 else None)
        elapsed_s = time.perf_counter() - generation_start_s

        _save_checkpoint(latest_path, "latest")
        _save_top_ranked()

        if trainer.state.best_reward > previous_best:
            _save_checkpoint(best_path, "best")
            previous_best = trainer.state.best_reward
        if trainer.state.best_single_reward > previous_best_single and trainer.top_rewards.shape[0] > 0:
            _save_payload(best_single_path, "best_single", _candidate_payload(0))
            previous_best_single = trainer.state.best_single_reward

        _log(
            f"gen={trainer.state.generation:6d}  "
            f"mean_reward={trainer.state.mean_reward:10.3f}  "
            f"top5_reward={trainer.state.episode_reward:10.3f}  "
            f"best_reward={trainer.state.best_reward:10.3f}  "
            f"elapsed={elapsed_s:7.2f}s"
        )

    total_elapsed_s = time.perf_counter() - start_time_s
    _save_checkpoint(latest_path, "latest")
    status = "interrupted" if interrupted else "completed"
    _log(f"Training {status} after {trainer.state.generation} generations in {total_elapsed_s:.2f}s")
    _log(f"Latest checkpoint: {latest_path}")
    _log(f"Best checkpoint:   {best_path}")
    _log(f"Best single:       {best_single_path}")
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
