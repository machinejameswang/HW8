"""Margin and support vectors page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import generate_dataset
from src.plotting import decision_boundary_figure
from src.svm_engine import compute_decision_mesh, train_svm
from src.theme import streamlit_css

st.set_page_config(page_title="Margin & Support Vectors", page_icon="2", layout="wide")
st.markdown(streamlit_css(), unsafe_allow_html=True)

st.title("2. Margin and Support Vectors")
st.markdown(
    """
    Support vectors are the samples closest to the decision boundary. They define
    the margin and have the strongest influence on the final SVM separator.
    """
)

dataset = generate_dataset("circles", n_samples=320, noise=0.1, random_state=42)
trained = train_svm(dataset.X, dataset.y, kernel="rbf", C=1.0, gamma=1.0, random_state=42)
mesh = compute_decision_mesh(trained, dataset.X, resolution=220)

col_plot, col_notes = st.columns([1.4, 0.8])
with col_plot:
    st.plotly_chart(decision_boundary_figure(dataset.X, dataset.y, trained, mesh), use_container_width=True)

with col_notes:
    st.metric("Support Vectors", len(trained.support_vectors))
    st.markdown(
        """
        ### What to notice
        - The white contour is `f(x)=0`, the decision boundary.
        - The purple contours are margin guides.
        - Amber open circles are support vectors.
        - Moving `C` and `gamma` changes how many points become support vectors.
        """
    )
    st.code("margin width = 2 / ||w||", language="text")

st.info("Next: tune the SVM yourself in the interactive page.")
