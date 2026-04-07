"""YAML/JSON runtime configuration loading helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .schema import DEFAULT_SPEC, RuntimeSpec, runtime_spec_from_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_runtime_spec(path: str | Path | None = None) -> RuntimeSpec:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError("Config file must end with .yaml, .yml, or .json")
    if not isinstance(data, dict):
        raise ValueError("Top-level config value must be a mapping/object.")
    return runtime_spec_from_dict(data)


def save_runtime_spec(path: str | Path, spec: RuntimeSpec) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        target_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    elif suffix == ".json":
        target_path.write_text(canonical_config_json(spec), encoding="utf-8")
    else:
        raise ValueError("Target config path must end with .yaml, .yml, or .json")
    return target_path


def canonical_config_json(spec: RuntimeSpec) -> str:
    return json.dumps(spec.to_dict(), indent=2, sort_keys=True)


def backend_agnostic_config_dict(spec_or_dict: RuntimeSpec | dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(spec_or_dict.to_dict() if isinstance(spec_or_dict, RuntimeSpec) else spec_or_dict)
    simulator = data.get("simulator")
    if isinstance(simulator, dict):
        simulator.pop("backend", None)
    return data


def canonical_backend_agnostic_config_json(spec: RuntimeSpec) -> str:
    return json.dumps(backend_agnostic_config_dict(spec), indent=2, sort_keys=True)


def config_json_matches_checkpoint(spec: RuntimeSpec, checkpoint_config_json: str) -> bool:
    try:
        checkpoint_data = json.loads(checkpoint_config_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(checkpoint_data, dict):
        return False
    active_json = json.dumps(backend_agnostic_config_dict(spec), sort_keys=True)
    checkpoint_json = json.dumps(backend_agnostic_config_dict(checkpoint_data), sort_keys=True)
    return active_json == checkpoint_json


def default_runtime_spec() -> RuntimeSpec:
    return runtime_spec_from_dict(DEFAULT_SPEC.to_dict())
