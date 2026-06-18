"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.svm_engine import TrainedSVM


@dataclass(frozen=True)
class MetricBundle:
    """Binary classification metrics for dashboard display."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    auc_score: float


def evaluate_model(trained: TrainedSVM) -> MetricBundle:
    """Evaluate the SVM on its held-out test set."""
    y_pred = trained.model.predict(trained.X_test)
    scores = trained.model.decision_function(trained.X_test)
    fpr, tpr, _ = roc_curve(trained.y_test, scores)

    return MetricBundle(
        accuracy=accuracy_score(trained.y_test, y_pred),
        precision=precision_score(trained.y_test, y_pred, zero_division=0),
        recall=recall_score(trained.y_test, y_pred, zero_division=0),
        f1=f1_score(trained.y_test, y_pred, zero_division=0),
        confusion=confusion_matrix(trained.y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        auc_score=auc(fpr, tpr),
    )
