"""Root landing page for the HW8 SVM learning app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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
    st.page_link("pages/1_SVM_Concept.py", label="Start Concept Lesson")
    st.page_link("pages/3_Interactive_SVM.py", label="Open Interactive SVM Lab")

st.divider()
st.subheader("Phase Outputs")

cards = st.columns(3)
with cards[0]:
    st.markdown("### Phase 1")
    st.caption("Manim concept animations")
    linear_video = ROOT / "outputs" / "phase1_LinearSVMMarginScene.mp4"
    kernel_video = ROOT / "outputs" / "phase1_KernelTrick3DScene.mp4"
    if linear_video.exists() and linear_video.stat().st_size > 0:
        st.video(str(linear_video))
    else:
        st.warning("Linear margin animation is missing. Run `./run_all_phases.ps1`.")
    if kernel_video.exists() and kernel_video.stat().st_size > 0:
        st.video(str(kernel_video))
    else:
        st.warning("Kernel trick animation is missing. Run `./run_all_phases.ps1`.")
with cards[1]:
    st.markdown("### Phase 2")
    st.caption("True Scikit-Learn SVM decision surface")
    decision_html = ROOT / "outputs" / "phase2_decision_boundary.html"
    if decision_html.exists() and decision_html.stat().st_size > 0:
        components.html(decision_html.read_text(encoding="utf-8"), height=460, scrolling=True)
    else:
        st.warning("Decision-boundary HTML is missing. Run `python scripts/export_phase2.py`.")
with cards[2]:
    st.markdown("### Phase 3")
    st.caption("Interactive Streamlit/Plotly dashboard")
    lift_html = ROOT / "outputs" / "phase2_kernel_lift_3d.html"
    if lift_html.exists() and lift_html.stat().st_size > 0:
        components.html(lift_html.read_text(encoding="utf-8"), height=460, scrolling=True)
    else:
        st.warning("3D kernel-lift HTML is missing. Run `python scripts/export_phase2.py`.")
    st.page_link("pages/3_Interactive_SVM.py", label="Go to Interactive Page")
