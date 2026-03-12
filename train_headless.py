"""Headless training entry point.

Runs ES training without starting the websocket server or React frontend and
saves periodic checkpoints to disk.
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import ai.jax_trainer as trainer_module
from ai.trainer import ESTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless quadruped training and save checkpoints.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to train.")
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
        help="Write a generation checkpoint every N generations.",
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

    if args.episode_seconds is not None:
        trainer_module.EPISODE_S = args.episode_seconds
    if args.population_size is not None:
        trainer_module.POP_SIZE = args.population_size

    trainer = ESTrainer(seed=args.seed)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    latest_path = args.out_dir / "latest.npz"
    best_path = args.out_dir / "best.npz"
    interrupted = False

    def _handle_interrupt(_sig, _frame) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    print("Headless training started")
    backend_description = f"JAX ES trainer ({getattr(trainer, 'backend', 'unknown')})"
    print(f"Backend: {backend_description}. No frontend or rendering is started.")
    if hasattr(trainer, "device_summary"):
        print(f"Devices: {trainer.device_summary}")
    print(
        f"Config: episode_s={trainer_module.EPISODE_S}  pop_size={trainer_module.POP_SIZE}  "
        f"save_every={args.save_every}"
    )
    start_time_s = time.perf_counter()
    previous_best = trainer.state.best_reward

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
                print(
                    f"gen={target_generation:6d}  "
                    f"step={step:6d}/{total_steps:6d}  "
                    f"reward={float(msg.get('reward', 0.0)):10.3f}  "
                    f"sim={float(msg.get('time_s', 0.0)):8.2f}s",
                    flush=True,
                )

        trainer.run_generation(on_step=_on_step if args.progress_every_steps > 0 else None)
        elapsed_s = time.perf_counter() - generation_start_s

        trainer.save_checkpoint(latest_path)
        should_save_generation = args.save_every > 0 and trainer.state.generation % args.save_every == 0
        if should_save_generation:
            trainer.save_checkpoint(args.out_dir / f"generation_{trainer.state.generation:06d}.npz")

        if trainer.state.best_reward > previous_best:
            trainer.save_checkpoint(best_path)
            previous_best = trainer.state.best_reward

        print(
            f"gen={trainer.state.generation:6d}  "
            f"mean_reward={trainer.state.mean_reward:10.3f}  "
            f"best_reward={trainer.state.best_reward:10.3f}  "
            f"elapsed={elapsed_s:7.2f}s"
        )

    total_elapsed_s = time.perf_counter() - start_time_s
    trainer.save_checkpoint(latest_path)
    status = "interrupted" if interrupted else "completed"
    print(f"Training {status} after {trainer.state.generation} generations in {total_elapsed_s:.2f}s")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint:   {best_path}")
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
