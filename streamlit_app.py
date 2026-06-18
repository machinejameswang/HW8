"""Root landing page for the HW8 SVM learning app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.theme import streamlit_css

ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="SVM Kernel Trick 3D",
    page_icon="SVM",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(streamlit_css(), unsafe_allow_html=True)

st.markdown('<div class="hero-title">SVM Kernel Trick 3D Learning Lab</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-subtitle">
    A 3-phase Support Vector Machine demo inspired by classroom Streamlit apps:
    concept videos, interactive SVM decision boundaries, kernel trick visualization,
    and a short quiz.
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1.15, 0.85])

with col_a:
    st.image(str(ROOT / "image.jpg"), use_container_width=True)

with col_b:
    st.markdown(
        """
        <div class="glass-panel">
        <h3>Learning Path</h3>
        <ol>
          <li><strong>SVM Concept</strong>: what a separating hyperplane means.</li>
          <li><strong>Margin & Support Vectors</strong>: why the closest points matter.</li>
          <li><strong>Interactive SVM</strong>: tune C, gamma, kernel, and datasets.</li>
          <li><strong>Kernel Trick</strong>: inspect 2D boundaries and 3D lift intuition.</li>
          <li><strong>Quiz</strong>: check the core ideas.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Use the sidebar page navigation to move through the lesson.")

st.divider()
st.subheader("Phase Outputs")

cards = st.columns(3)
with cards[0]:
    st.markdown("### Phase 1")
    st.caption("Manim concept animations")
    if (ROOT / "outputs" / "phase1_KernelTrick3DScene.mp4").exists():
        st.video(str(ROOT / "outputs" / "phase1_KernelTrick3DScene.mp4"))
with cards[1]:
    st.markdown("### Phase 2")
    st.caption("True Scikit-Learn SVM decision surface")
    st.link_button("Open 2D Decision Boundary", "outputs/phase2_decision_boundary.html")
with cards[2]:
    st.markdown("### Phase 3")
    st.caption("Interactive Streamlit/Plotly dashboard")
    st.link_button("Go to Interactive Page", "/Interactive_SVM")
