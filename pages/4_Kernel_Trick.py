"""Kernel trick teaching page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import generate_dataset
from src.plotting import decision_boundary_figure, kernel_lift_figure
from src.svm_engine import compute_decision_mesh, train_svm
from src.theme import streamlit_css

st.set_page_config(page_title="Kernel Trick", page_icon="4", layout="wide")
st.markdown(streamlit_css(), unsafe_allow_html=True)

st.title("4. Kernel Trick")
st.markdown(
    """
    The kernel trick lets SVM build nonlinear decision boundaries without
    explicitly constructing every high-dimensional feature. The 3D lift shown
    here is educational; the real RBF kernel is implicit.
    """
)

video = ROOT / "outputs" / "phase1_KernelTrick3DScene.mp4"
if video.exists() and video.stat().st_size > 0:
    st.video(str(video))
else:
    st.info("Kernel trick animation is not available yet. The interactive SVM plots below still use the live Scikit-Learn model.")

st.divider()
st.subheader("Compare kernels on blue-core / red-ring data")

kernel_label = st.radio("Kernel", ["linear", "rbf", "poly", "sigmoid"], horizontal=True, index=1)
C = st.select_slider("C", options=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], value=1.0)
gamma = st.slider("gamma", 0.05, 5.0, 1.0, 0.05)
degree = st.slider("degree", 2, 6, 3)

dataset = generate_dataset("circles", n_samples=360, noise=0.12, random_state=7)
trained = train_svm(dataset.X, dataset.y, kernel=kernel_label, C=C, gamma=gamma, degree=degree, random_state=7)
mesh = compute_decision_mesh(trained, dataset.X, resolution=240)

col_2d, col_3d = st.columns(2)
with col_2d:
    st.plotly_chart(decision_boundary_figure(dataset.X, dataset.y, trained, mesh), use_container_width=True)
with col_3d:
    st.plotly_chart(kernel_lift_figure(dataset.X, dataset.y, trained, mesh), use_container_width=True)

st.caption("For RBF lift, center blue points should have larger z values than red outer-ring points.")
