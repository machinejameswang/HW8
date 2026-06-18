"""Export Phase 2 SVM computation artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import generate_dataset
from src.metrics import evaluate_model
from src.plotting import decision_boundary_figure, kernel_lift_figure
from src.svm_engine import compute_decision_mesh, kernel_lift_z, train_svm


def main() -> None:
    """Train the default RBF SVM and export interactive HTML artifacts."""
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    dataset = generate_dataset("circles", n_samples=400, noise=0.12, random_state=42)
    trained = train_svm(
        dataset.X,
        dataset.y,
        kernel="rbf",
        C=1.0,
        gamma=1.0,
        degree=3,
        test_size=0.3,
        random_state=42,
    )
    mesh = compute_decision_mesh(trained, dataset.X, resolution=260)
    metrics = evaluate_model(trained)
    lifted_z = kernel_lift_z(dataset.X, kernel="rbf", gamma=1.0)

    decision_fig = decision_boundary_figure(dataset.X, dataset.y, trained, mesh)
    lift_fig = kernel_lift_figure(dataset.X, dataset.y, trained, mesh)
    decision_fig.write_html(output_dir / "phase2_decision_boundary.html", include_plotlyjs="cdn")
    lift_fig.write_html(output_dir / "phase2_kernel_lift_3d.html", include_plotlyjs="cdn")

    radii = (dataset.X[:, 0] ** 2 + dataset.X[:, 1] ** 2) ** 0.5
    summary = {
        "dataset": dataset.label,
        "description": dataset.description,
        "n_samples": int(len(dataset.y)),
        "kernel": trained.kernel,
        "C": trained.C,
        "gamma": trained.gamma,
        "support_vector_count": int(len(trained.support_vectors)),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "auc": float(metrics.auc_score),
        "blue_core_mean_radius": float(radii[dataset.y == 0].mean()),
        "red_outer_mean_radius": float(radii[dataset.y == 1].mean()),
        "blue_core_mean_lift_z": float(lifted_z[dataset.y == 0].mean()),
        "red_outer_mean_lift_z": float(lifted_z[dataset.y == 1].mean()),
        "label_mapping_ok": bool(radii[dataset.y == 0].mean() < radii[dataset.y == 1].mean()),
        "rbf_lift_order_ok": bool(lifted_z[dataset.y == 0].mean() > lifted_z[dataset.y == 1].mean()),
    }
    (output_dir / "phase2_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
