"""Dataset generation utilities for SVM demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons


@dataclass(frozen=True)
class DatasetBundle:
    """Container for a generated 2D classification dataset."""

    X: np.ndarray
    y: np.ndarray
    label: str
    description: str


DATASET_LABELS = {
    "circles": "Blue Core / Red Ring",
    "moons": "Interlocking Moons",
    "linear": "Linear Separable",
    "xor": "XOR Quadrants",
}


def generate_dataset(
    dataset: str,
    n_samples: int = 400,
    noise: float = 0.15,
    random_state: int = 42,
) -> DatasetBundle:
    """Generate a 2D binary dataset for SVM visualization."""
    rng = np.random.default_rng(random_state)
    dataset = dataset.lower()

    if dataset == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.42,
            random_state=random_state,
        )
        # sklearn returns y=0 for the outer ring and y=1 for the inner ring.
        # Remap so the whole project consistently uses 0=blue core, 1=red ring.
        y = 1 - y
        description = (
            "Blue points form the inner core and red points form the outer ring, "
            "ideal for demonstrating the RBF kernel lift."
        )
    elif dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        description = "Two interlocking moon-shaped classes for demonstrating nonlinear decision boundaries."
    elif dataset == "linear":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.8,
            flip_y=min(noise * 0.45, 0.18),
            random_state=random_state,
        )
        description = "A mostly linearly separable dataset for comparing linear and nonlinear kernels."
    elif dataset == "xor":
        X = rng.uniform(-1.4, 1.4, size=(n_samples, 2))
        X += rng.normal(0.0, noise, size=X.shape)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        description = "An XOR quadrant dataset that requires a nonlinear kernel for clean separation."
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DatasetBundle(
        X=X.astype(float),
        y=y.astype(int),
        label=DATASET_LABELS.get(dataset, dataset.title()),
        description=description,
    )
