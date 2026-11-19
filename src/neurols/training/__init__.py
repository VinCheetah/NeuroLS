"""Training and neuroevolution algorithms for NeuroLS."""

from neurols.training.base import BaseOptimizer, GenerationRecord, TrainResult
from neurols.training.cma_es import CMAESOptimizer
from neurols.training.registry import (
    ModelManifest,
    delete_saved_model,
    list_saved_models,
    load_model_checkpoint,
    resolve_model_path,
    save_model_checkpoint,
)

__all__ = [
    "BaseOptimizer",
    "GenerationRecord",
    "TrainResult",
    "CMAESOptimizer",
    "ModelManifest",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "list_saved_models",
    "resolve_model_path",
    "delete_saved_model",
]
