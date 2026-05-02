"""Model catalog for trainable quadruped policies."""

from .lab import NotebookModel, apply_notebook_model, register_notebook_model, write_policy_module
from .registry import (
    CURRENT_MODEL_TYPE,
    DEFAULT_MODEL_REGISTRY_PATH,
    ModelDefinition,
    PolicyPlugin,
    get_model_definition,
    list_model_definitions,
    load_model_definitions,
    load_policy_plugin,
    model_registry_path,
    refresh_model_definitions,
    register_inline_policy_factory,
    register_model_definition,
    validate_model_definition,
)

__all__ = [
    "CURRENT_MODEL_TYPE",
    "DEFAULT_MODEL_REGISTRY_PATH",
    "ModelDefinition",
    "NotebookModel",
    "PolicyPlugin",
    "apply_notebook_model",
    "get_model_definition",
    "list_model_definitions",
    "load_model_definitions",
    "load_policy_plugin",
    "model_registry_path",
    "refresh_model_definitions",
    "register_inline_policy_factory",
    "register_model_definition",
    "register_notebook_model",
    "validate_model_definition",
    "write_policy_module",
]
