_TRAINER_EXPORTS = (
    "ESTrainer",
    "JaxESTrainer",
    "TrainingState",
    "apply_runtime_spec",
    "current_runtime_spec",
)
__all__ = list(_TRAINER_EXPORTS)


def __getattr__(name: str):
    if name in _TRAINER_EXPORTS:
        from . import jax_trainer

        return getattr(jax_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
