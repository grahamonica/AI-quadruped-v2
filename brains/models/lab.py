"""Notebook-first helpers for registering and testing model variants.

This API keeps model registration, optional plugin parameter inference, and
runtime-spec updates in one place so notebook-defined brains can be dropped into
the repo and tested by `train_headless.py` and the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax
from jax.flatten_util import ravel_pytree

from brains.config import DEFAULT_SPEC, RuntimeSpec
from .registry import ModelDefinition, load_policy_plugin, register_model_definition


@dataclass(frozen=True)
class NotebookModel:
    model_type: str
    architecture: str
    description: str
    input_size: int
    output_size: int
    parameter_count: int | None = None
    trainer: str = str(DEFAULT_SPEC.model.trainer)
    policy_entrypoint: str | None = None
    control_mode: str = str(DEFAULT_SPEC.control.mode)
    command_vocabulary: tuple[str, ...] = DEFAULT_SPEC.control.command_vocabulary
    default_command_speed: float = float(DEFAULT_SPEC.control.default_command_speed)
    command_update_interval_s: float = float(DEFAULT_SPEC.control.command_update_interval_s)
    command_default_duration_s: float = float(DEFAULT_SPEC.control.command_default_duration_s)
    command_max_duration_s: float = float(DEFAULT_SPEC.control.command_max_duration_s)
    positional_encoding: str = str(DEFAULT_SPEC.model.positional_encoding)
    positional_encoding_gain: float = float(DEFAULT_SPEC.model.positional_encoding_gain)


def register_notebook_model(
    model: NotebookModel,
    *,
    spec: RuntimeSpec | None = None,
    persist: bool = True,
    registry_path: str | Path | None = None,
) -> ModelDefinition:
    parameter_count = model.parameter_count
    if parameter_count is None and model.policy_entrypoint is not None:
        if spec is None:
            raise ValueError("register_notebook_model requires spec when inferring parameter_count from policy_entrypoint.")
        probe_definition = ModelDefinition(
            type=model.model_type,
            architecture=model.architecture,
            trainer=model.trainer,
            input_size=model.input_size,
            output_size=model.output_size,
            parameter_count=1,
            description=model.description,
            policy_entrypoint=model.policy_entrypoint,
        )
        plugin = load_policy_plugin(spec, probe_definition)
        if plugin is None:
            raise ValueError("Could not load policy plugin for parameter count inference.")
        probe_params = plugin.init_params(jax.random.PRNGKey(0))
        probe_flat, _ = ravel_pytree(probe_params)
        parameter_count = int(probe_flat.shape[0])
    if parameter_count is None:
        raise ValueError("NotebookModel.parameter_count must be provided when no policy_entrypoint is set.")

    definition = ModelDefinition(
        type=model.model_type,
        architecture=model.architecture,
        trainer=model.trainer,
        input_size=model.input_size,
        output_size=model.output_size,
        parameter_count=int(parameter_count),
        description=model.description,
        policy_entrypoint=model.policy_entrypoint,
    )
    return register_model_definition(definition, persist=persist, registry_path=registry_path)


def apply_notebook_model(spec: RuntimeSpec, model: NotebookModel) -> RuntimeSpec:
    updated = replace(
        spec,
        model=replace(
            spec.model,
            type=model.model_type,
            architecture=model.architecture,
            trainer=model.trainer,
            description=model.description,
            positional_encoding=model.positional_encoding,
            positional_encoding_gain=model.positional_encoding_gain,
        ),
        control=replace(
            spec.control,
            mode=model.control_mode,
            command_vocabulary=model.command_vocabulary,
            default_command_speed=model.default_command_speed,
            command_update_interval_s=model.command_update_interval_s,
            command_default_duration_s=model.command_default_duration_s,
            command_max_duration_s=model.command_max_duration_s,
        ),
    )
    updated.validate()
    return updated


def write_policy_module(module_path: str | Path, source: str) -> Path:
    path = Path(module_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path
