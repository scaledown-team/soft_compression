"""Evaluation utilities for ScaleDown."""

from .evaluator import ScaleDownEvaluator, evaluate_model
from .metrics import compute_rag_metrics, plot_training_curves

__all__ = [
    "ScaleDownEvaluator",
    "evaluate_model",
    "compute_rag_metrics",
    "plot_training_curves",
]
