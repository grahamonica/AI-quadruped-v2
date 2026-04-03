"""Structured logging and run artifact management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.config import RuntimeSpec, canonical_config_json, save_runtime_spec


class JsonLineFormatter(logging.Formatter):
    """Simple JSONL formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    config_path: Path
    events_path: Path
    metrics_path: Path
    quality_report_path: Path


class MetricsSink:
    """Append-only JSONL metrics file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **payload: Any) -> None:
        record = {"kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "run"


def create_run_artifacts(spec: RuntimeSpec, run_name: str | None = None) -> RunArtifacts:
    now = datetime.now(tz=timezone.utc)
    name = _sanitize_name(run_name or spec.name)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{name}"
    run_dir = Path(spec.logging.root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_runtime_spec(run_dir / "resolved_config.yaml", spec)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        config_path=config_path,
        events_path=run_dir / spec.logging.events_filename,
        metrics_path=run_dir / spec.logging.metrics_filename,
        quality_report_path=run_dir / "quality_report.json",
    )


def configure_logging(spec: RuntimeSpec, artifacts: RunArtifacts) -> logging.Logger:
    logger = logging.getLogger("quadruped")
    logger.setLevel(getattr(logging, spec.logging.level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(artifacts.events_path, encoding="utf-8")
    file_handler.setFormatter(JsonLineFormatter())
    logger.addHandler(file_handler)

    if spec.logging.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(console_handler)

    logger.info(
        "Run initialized",
        extra={
            "run_id": artifacts.run_id,
            "run_dir": str(artifacts.run_dir),
            "config_path": str(artifacts.config_path),
            "config_json": canonical_config_json(spec),
        },
    )
    return logger


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
