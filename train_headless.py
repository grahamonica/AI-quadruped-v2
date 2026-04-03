"""Headless training entry point with config-driven runtime validation."""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import ai.jax_trainer as trainer_module
from ai.config import DEFAULT_CONFIG_PATH, RuntimeSpec, load_runtime_spec
from ai.infra import MetricsSink, configure_logging, create_run_artifacts, write_json
from ai.quality import QualityGateRunner
from ai.trainer import ESTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless quadruped training and save checkpoints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML or JSON runtime spec. Defaults to configs/default.yaml.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name used for log artifact directories.")
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
        help="Retained for CLI compatibility; numbered generation checkpoints are not written.",
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
        help="Override episode.episode_s from the runtime spec.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override training.population_size from the runtime spec.",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=0,
        help="If > 0, emit step progress every N streamed episode steps.",
    )
    parser.add_argument(
        "--skip-quality-gates",
        action="store_true",
        help="Skip fast runtime validation before training starts.",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Run the quality gates, write the report, and exit without training.",
    )
    return parser.parse_args()


def _apply_cli_overrides(spec: RuntimeSpec, args: argparse.Namespace) -> RuntimeSpec:
    updated = spec
    if args.episode_seconds is not None:
        updated = replace(updated, episode=replace(updated.episode, episode_s=float(args.episode_seconds)))
    if args.population_size is not None:
        updated = replace(updated, training=replace(updated.training, population_size=int(args.population_size)))
    updated.validate()
    return updated


def main() -> int:
    args = _parse_args()
    spec = _apply_cli_overrides(load_runtime_spec(args.config), args)
    trainer_module.apply_runtime_spec(spec)

    artifacts = create_run_artifacts(spec, run_name=args.run_name)
    logger = configure_logging(spec, artifacts)
    metrics = MetricsSink(artifacts.metrics_path)
    metrics.emit(
        "run_started",
        config_path=str(Path(args.config).resolve()),
        config_name=spec.name,
        generations=args.generations,
        seed=args.seed,
        checkpoint_dir=str(args.out_dir.resolve()),
    )

    quality_report = None
    if spec.quality_gates.enabled and spec.quality_gates.run_on_startup and not args.skip_quality_gates:
        quality_report = QualityGateRunner(spec).run()
        write_json(artifacts.quality_report_path, quality_report.to_dict())
        logger.info("Quality gates completed", extra={"quality_passed": quality_report.passed, "quality_report": quality_report.to_dict()})
        metrics.emit("quality_gates", passed=quality_report.passed, report=quality_report.to_dict())
        if not quality_report.passed:
            logger.error("Quality gates failed; refusing to start training.", extra={"quality_report_path": str(artifacts.quality_report_path)})
            return 2
    elif args.skip_quality_gates:
        logger.warning("Quality gates skipped by CLI flag.")
        metrics.emit("quality_gates_skipped")

    if args.quality_only:
        logger.info("Exiting after quality-only run.", extra={"quality_report_path": str(artifacts.quality_report_path)})
        return 0

    start_time_s = time.perf_counter()
    trainer = ESTrainer(seed=args.seed, spec=spec)
    latest_path = args.out_dir / "latest.npz"

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        logger.info("Resumed from checkpoint", extra={"checkpoint_path": str(args.resume)})
    elif latest_path.exists():
        trainer.load_checkpoint(latest_path)
        logger.info("Auto-resumed from latest checkpoint", extra={"checkpoint_path": str(latest_path)})
    elif (args.out_dir / "best.npz").exists():
        fallback_best = args.out_dir / "best.npz"
        trainer.load_checkpoint(fallback_best)
        logger.info("Auto-resumed from best checkpoint", extra={"checkpoint_path": str(fallback_best)})

    best_path = args.out_dir / "best.npz"
    top_paths = [args.out_dir / f"top_{rank:02d}.npz" for rank in range(1, trainer_module.PARENT_ELITE_COUNT + 1)]
    best_single_path = args.out_dir / "best_single.npz"
    interrupted = False

    def _save_checkpoint(path: Path, label: str) -> Path:
        saved_path = trainer.save_checkpoint(path)
        logger.info(
            "Checkpoint updated",
            extra={
                "label": label,
                "generation": trainer.state.generation,
                "checkpoint_path": str(saved_path),
            },
        )
        return saved_path

    def _save_payload(path: Path, label: str, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        logger.info(
            "Checkpoint updated",
            extra={
                "label": label,
                "generation": trainer.state.generation,
                "checkpoint_path": str(path),
            },
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

    logger.info(
        "Headless training started",
        extra={
            "backend": trainer.backend,
            "devices": trainer.device_summary,
            "population_size": trainer_module.POP_SIZE,
            "episode_seconds": trainer_module.EPISODE_S,
            "run_dir": str(artifacts.run_dir),
        },
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
                logger.info(
                    "Swarm progress",
                    extra={
                        "generation": target_generation,
                        "step": step,
                        "total_steps": total_steps,
                        "reward": float(msg.get("reward", 0.0)),
                        "time_s": float(msg.get("time_s", 0.0)),
                    },
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

        metrics.emit(
            "generation",
            generation=trainer.state.generation,
            mean_reward=trainer.state.mean_reward,
            top_reward=float(trainer.top_rewards[0]) if trainer.top_rewards.size else None,
            best_reward=trainer.state.best_reward,
            best_single_reward=trainer.state.best_single_reward,
            elapsed_s=elapsed_s,
            goal_xyz=list(trainer.state.goal_xyz),
        )
        logger.info(
            "Generation completed",
            extra={
                "generation": trainer.state.generation,
                "mean_reward": trainer.state.mean_reward,
                "top5_reward": trainer.state.episode_reward,
                "best_reward": trainer.state.best_reward,
                "elapsed_s": elapsed_s,
            },
        )

    total_elapsed_s = time.perf_counter() - start_time_s
    _save_checkpoint(latest_path, "latest")
    status = "interrupted" if interrupted else "completed"
    summary = {
        "status": status,
        "generation": trainer.state.generation,
        "elapsed_s": total_elapsed_s,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "best_single_checkpoint": str(best_single_path),
        "quality_report_path": str(artifacts.quality_report_path) if quality_report is not None else None,
    }
    write_json(artifacts.run_dir / "summary.json", summary)
    metrics.emit("run_finished", **summary)
    logger.info("Training finished", extra=summary)
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
