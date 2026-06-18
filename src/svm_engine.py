"""Scikit-Learn SVM training and decision-surface computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class TrainedSVM:
    """A trained SVM bundle with original-space data and support vectors."""

    model: Pipeline
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    support_vectors: np.ndarray
    support_indices: np.ndarray
    kernel: str
    C: float
    gamma: float | str
    degree: int


@dataclass(frozen=True)
class DecisionMesh:
    """Dense mesh containing raw SVM decision scores and predictions."""

    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray
    predictions: np.ndarray


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float = 1.0,
    degree: int = 3,
    test_size: float = 0.3,
    random_state: int = 42,
) -> TrainedSVM:
    """Train a scaled binary SVM and return original-space support vectors."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma if kernel != "linear" else "scale",
        degree=degree,
        coef0=1.0,
    )
    model = Pipeline([("scaler", StandardScaler()), ("svc", svc)])
    model.fit(X_train, y_train)

    scaler = model.named_steps["scaler"]
    fitted_svc = model.named_steps["svc"]
    support_vectors = scaler.inverse_transform(fitted_svc.support_vectors_)

    return TrainedSVM(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        support_vectors=support_vectors,
        support_indices=fitted_svc.support_,
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
    )


def compute_decision_mesh(
    trained: TrainedSVM,
    X: np.ndarray,
    resolution: int = 300,
    padding: float = 0.75,
) -> DecisionMesh:
    """Compute a dense 2D decision mesh using the trained SVM decision function."""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = trained.model.decision_function(grid).reshape(xx.shape)
    predictions = trained.model.predict(grid).reshape(xx.shape)
    return DecisionMesh(xx=xx, yy=yy, zz=zz, predictions=predictions)


def kernel_lift_z(
    X: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Return a didactic 3D lift coordinate for a selected kernel family."""
    x = X[:, 0]
    y = X[:, 1]

    if kernel == "rbf":
        return np.exp(-gamma * (x**2 + y**2))
    if kernel == "linear":
        return x + y
    if kernel == "poly":
        return (gamma * (x + y) + coef0) ** degree
    if kernel == "sigmoid":
        return np.tanh(gamma * (x + y) + coef0)
    raise ValueError(f"Unknown kernel: {kernel}")


def lifted_surface(
    mesh: DecisionMesh,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
) -> np.ndarray:
    """Compute the didactic z surface for the full mesh."""
    flat = np.c_[mesh.xx.ravel(), mesh.yy.ravel()]
    return kernel_lift_z(flat, kernel=kernel, gamma=gamma, degree=degree).reshape(mesh.xx.shape)
