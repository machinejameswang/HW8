"""Tests for SVM data generation and decision-surface computation."""

from __future__ import annotations

from src.data import generate_dataset
from src.metrics import evaluate_model
from src.svm_engine import compute_decision_mesh, train_svm


def test_dataset_generation_shape() -> None:
    bundle = generate_dataset("circles", n_samples=240, noise=0.12, random_state=7)
    assert bundle.X.shape == (240, 2)
    assert bundle.y.shape == (240,)
    assert set(bundle.y) == {0, 1}


def test_circles_label_mapping_matches_visual_story() -> None:
    bundle = generate_dataset("circles", n_samples=300, noise=0.0, random_state=42)
    radii = (bundle.X[:, 0] ** 2 + bundle.X[:, 1] ** 2) ** 0.5
    assert radii[bundle.y == 0].mean() < radii[bundle.y == 1].mean()


def test_svm_training_support_vectors() -> None:
    bundle = generate_dataset("circles", n_samples=240, noise=0.08, random_state=7)
    trained = train_svm(bundle.X, bundle.y, kernel="rbf", C=1.0, gamma=1.2, random_state=7)
    assert 0 < len(trained.support_vectors) <= len(bundle.X)


def test_decision_mesh_shape() -> None:
    bundle = generate_dataset("moons", n_samples=220, noise=0.14, random_state=11)
    trained = train_svm(bundle.X, bundle.y, kernel="rbf", C=1.0, gamma=1.0, random_state=11)
    mesh = compute_decision_mesh(trained, bundle.X, resolution=120)
    assert mesh.xx.shape == (120, 120)
    assert mesh.yy.shape == (120, 120)
    assert mesh.zz.shape == (120, 120)
    assert mesh.predictions.shape == (120, 120)


def test_metrics_are_valid() -> None:
    bundle = generate_dataset("xor", n_samples=260, noise=0.08, random_state=19)
    trained = train_svm(bundle.X, bundle.y, kernel="rbf", C=2.0, gamma=1.5, random_state=19)
    metrics = evaluate_model(trained)
    for value in [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1, metrics.auc_score]:
        assert 0.0 <= value <= 1.0


def test_rbf_circles_reasonable_accuracy() -> None:
    bundle = generate_dataset("circles", n_samples=400, noise=0.1, random_state=42)
    trained = train_svm(bundle.X, bundle.y, kernel="rbf", C=1.0, gamma=1.0, random_state=42)
    metrics = evaluate_model(trained)
    assert metrics.accuracy > 0.8
