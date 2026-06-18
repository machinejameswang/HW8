"""Concept page for the SVM learning app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.theme import streamlit_css

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="SVM Concept", page_icon="1", layout="wide")
st.markdown(streamlit_css(), unsafe_allow_html=True)

st.title("1. SVM Concept")
st.markdown(
    """
    Support Vector Machine finds a separating boundary that leaves the widest
    possible margin between two classes. In 2D this boundary is a line; in higher
    dimensions it is a hyperplane.
    """
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Core idea")
    st.markdown(
        """
        - The classifier separates classes with a hyperplane.
        - The margin is the distance from the hyperplane to the closest samples.
        - The closest samples are the support vectors.
        - SVM prefers the separator with the largest margin.
        """
    )
    st.code("decision boundary: w^T x + b = 0\nmargins: w^T x + b = +1 and -1", language="text")

with right:
    st.subheader("Concept animation")
    video = ROOT / "outputs" / "phase1_LinearSVMMarginScene.mp4"
    if video.exists():
        st.video(str(video))
    else:
        st.info("Run `./run_all_phases.ps1` to generate the Manim video.")

st.divider()
st.success("Next: learn why support vectors define the margin.")
