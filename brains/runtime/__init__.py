"""Runtime helpers shared by the headless trainer, viewer, and tests."""

from importlib import import_module
from typing import Any

from .logging import MetricsSink, configure_logging, create_run_artifacts, write_json
from .model_store import (
    checkpoint_matches_spec,
    create_model_run_paths,
    discover_model_artifacts,
    find_model_artifact,
    resolve_viewer_checkpoint,
    runtime_spec_from_checkpoint,
    viewer_checkpoint_candidates,
    write_model_manifest,
)

_LAZY_EXPORTS = {
    "DiscreteLegPlayback": (".playback", "DiscreteLegPlayback"),
    "EsTrainerPlayback": (".playback", "EsTrainerPlayback"),
    "Playback": (".playback", "Playback"),
    "PlaybackLoadResult": (".playback", "PlaybackLoadResult"),
    "PlaybackState": (".playback", "PlaybackState"),
    "load_playback_for_selection": (".playback", "load_playback_for_selection"),
    "QualityGateRunner": (".quality_gates", "QualityGateRunner"),
    "collect_regression_metrics": (".quality_gates", "collect_regression_metrics"),
    "compare_regression_to_baseline": (".quality_gates", "compare_regression_to_baseline"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "DiscreteLegPlayback",
    "EsTrainerPlayback",
    "MetricsSink",
    "Playback",
    "PlaybackLoadResult",
    "PlaybackState",
    "QualityGateRunner",
    "collect_regression_metrics",
    "compare_regression_to_baseline",
    "checkpoint_matches_spec",
    "configure_logging",
    "create_model_run_paths",
    "create_run_artifacts",
    "discover_model_artifacts",
    "find_model_artifact",
    "load_playback_for_selection",
    "resolve_viewer_checkpoint",
    "runtime_spec_from_checkpoint",
    "viewer_checkpoint_candidates",
    "write_json",
    "write_model_manifest",
]
