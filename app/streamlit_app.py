"""Streamlit entrypoint for the SVM interactive arena."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import generate_dataset
from src.metrics import evaluate_model
from src.plotting import (
    confusion_matrix_figure,
    decision_boundary_figure,
    kernel_lift_figure,
    roc_curve_figure,
)
from src.svm_engine import compute_decision_mesh, train_svm
from src.theme import COLORS, streamlit_css


DATASET_OPTIONS = {
    "Blue Core / Red Ring": "circles",
    "Interlocking Moons": "moons",
    "Linear Separable": "linear",
    "XOR Quadrants": "xor",
}

KERNEL_OPTIONS = {
    "RBF Gaussian": "rbf",
    "Linear": "linear",
    "Polynomial": "poly",
    "Sigmoid": "sigmoid",
}


@st.cache_data(show_spinner=False)
def cached_dataset(dataset: str, n_samples: int, noise: float, random_state: int):
    """Cache generated datasets across Streamlit reruns."""
    return generate_dataset(dataset, n_samples, noise, random_state)


@st.cache_data(show_spinner=False)
def cached_training(
    X,
    y,
    kernel: str,
    C: float,
    gamma: float,
    degree: int,
    test_size: float,
    random_state: int,
    resolution: int,
):
    """Cache SVM training, decision mesh, and metrics."""
    trained = train_svm(
        X,
        y,
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        test_size=test_size,
        random_state=random_state,
    )
    mesh = compute_decision_mesh(trained, X, resolution=resolution)
    metrics = evaluate_model(trained)
    return trained, mesh, metrics


def render_sidebar() -> dict:
    """Render controls and return selected configuration."""
    st.sidebar.title("SVM Control Panel")
    st.sidebar.caption("Tune the dataset, kernel, and margin parameters to inspect SVM geometry.")

    dataset_label = st.sidebar.selectbox("Dataset Geometry", list(DATASET_OPTIONS), index=0)
    kernel_label = st.sidebar.selectbox("Kernel Function", list(KERNEL_OPTIONS), index=0)

    n_samples = st.sidebar.slider("Samples", 100, 1000, 400, 50)
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.15, 0.01)
    C = st.sidebar.slider("Regularization C", 0.01, 100.0, 1.0, 0.01, format="%.2f")

    kernel = KERNEL_OPTIONS[kernel_label]
    gamma = 1.0
    degree = 3
    if kernel in {"rbf", "poly", "sigmoid"}:
        gamma = st.sidebar.slider("Gamma", 0.01, 10.0, 1.0, 0.01, format="%.2f")
    if kernel == "poly":
        degree = st.sidebar.slider("Polynomial Degree", 2, 6, 3, 1)

    resolution = st.sidebar.slider("Mesh Resolution", 120, 420, 260, 20)
    test_size = st.sidebar.slider("Test Split", 0.15, 0.5, 0.3, 0.05)
    random_state = st.sidebar.number_input("Random Seed", value=42, step=1)

    return {
        "dataset": DATASET_OPTIONS[dataset_label],
        "dataset_label": dataset_label,
        "kernel": kernel,
        "kernel_label": kernel_label,
        "n_samples": n_samples,
        "noise": noise,
        "C": C,
        "gamma": gamma,
        "degree": degree,
        "resolution": resolution,
        "test_size": test_size,
        "random_state": int(random_state),
    }


def metric_row(metrics, support_count: int) -> None:
    """Render the top metric strip."""
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics.accuracy:.3f}")
    cols[1].metric("Precision", f"{metrics.precision:.3f}")
    cols[2].metric("Recall", f"{metrics.recall:.3f}")
    cols[3].metric("F1 Score", f"{metrics.f1:.3f}")
    cols[4].metric("Support Vectors", f"{support_count}")


def concept_notes(config: dict) -> None:
    """Render compact SVM concept notes."""
    st.markdown(
        f"""
        <div class="glass-panel">
        <h3>Kernel intuition</h3>
        <p>
        Current kernel: <strong>{config["kernel_label"]}</strong>. The SVM decision function
        is determined by support vectors:
        </p>
        <pre>f(x) = sum_i alpha_i y_i K(x_i, x) + b</pre>
        <p>The RBF Gaussian kernel measures similarity with distance:</p>
        <pre>K(x_i, x) = exp(-gamma * ||x_i - x||^2)</pre>
        <p>
        <strong>C</strong> controls the penalty for margin violations.
        <strong>gamma</strong> controls how local the RBF influence is.
        Larger gamma values create more flexible local boundaries.
        </p>
        <p>
        The 3D lift is an educational visualization. The true RBF feature space is
        implicit and can be high-dimensional or infinite-dimensional.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_animation_tab() -> None:
    """Render generated Manim animation videos when available."""
    animation_files = [
        ("Linear SVM Margin", ROOT / "outputs" / "phase1_LinearSVMMarginScene.mp4"),
        ("Kernel Trick 3D", ROOT / "outputs" / "phase1_KernelTrick3DScene.mp4"),
    ]
    cols = st.columns(2)
    for col, (title, path) in zip(cols, animation_files):
        with col:
            st.subheader(title)
            if path.exists():
                st.video(str(path))
            else:
                st.info("Animation not found. Run `./run_all_phases.ps1` to render Phase 1.")


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="SVM Interactive Arena",
        page_icon="SVM",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(streamlit_css(), unsafe_allow_html=True)

    config = render_sidebar()
    dataset = cached_dataset(
        config["dataset"],
        config["n_samples"],
        config["noise"],
        config["random_state"],
    )

    with st.spinner("Training SVM and computing decision surface..."):
        trained, mesh, metrics = cached_training(
            dataset.X,
            dataset.y,
            config["kernel"],
            config["C"],
            config["gamma"],
            config["degree"],
            config["test_size"],
            config["random_state"],
            config["resolution"],
        )

    st.markdown('<div class="hero-title">Support Vector Machine Interactive Arena</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-subtitle">{dataset.description} '
        f'Current kernel: <span style="color:{COLORS["cyan"]};">{config["kernel_label"]}</span>.</div>',
        unsafe_allow_html=True,
    )

    metric_row(metrics, len(trained.support_vectors))

    tab_decision, tab_lift, tab_animation, tab_metrics, tab_notes = st.tabs(
        ["2D Decision Space", "3D Kernel Lift", "Manim Animations", "Model Diagnostics", "Concept Notes"]
    )

    with tab_decision:
        fig = decision_boundary_figure(dataset.X, dataset.y, trained, mesh)
        st.plotly_chart(fig, use_container_width=True)

    with tab_lift:
        fig = kernel_lift_figure(dataset.X, dataset.y, trained, mesh)
        st.plotly_chart(fig, use_container_width=True)

    with tab_animation:
        render_animation_tab()

    with tab_metrics:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(confusion_matrix_figure(metrics), use_container_width=True)
        with col_b:
            st.plotly_chart(roc_curve_figure(metrics), use_container_width=True)

    with tab_notes:
        concept_notes(config)


if __name__ == "__main__":
    main()
