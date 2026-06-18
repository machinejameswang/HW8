"""Short SVM quiz page."""

from __future__ import annotations

import streamlit as st

from src.theme import streamlit_css

st.set_page_config(page_title="SVM Quiz", page_icon="5", layout="wide")
st.markdown(streamlit_css(), unsafe_allow_html=True)

st.title("5. Quiz")
st.markdown("Check the main SVM ideas from the lesson.")

questions = [
    {
        "q": "What are support vectors?",
        "options": [
            "The closest training samples to the decision boundary",
            "Random samples removed before training",
            "The farthest samples from both classes",
        ],
        "answer": "The closest training samples to the decision boundary",
    },
    {
        "q": "What does C control?",
        "options": [
            "Penalty for margin violations",
            "Number of features",
            "Plotly camera angle",
        ],
        "answer": "Penalty for margin violations",
    },
    {
        "q": "What does a larger RBF gamma usually do?",
        "options": [
            "Makes the decision boundary more local and flexible",
            "Forces a purely linear boundary",
            "Deletes support vectors",
        ],
        "answer": "Makes the decision boundary more local and flexible",
    },
    {
        "q": "Is the educational 3D lift the full true RBF feature space?",
        "options": [
            "No, true RBF SVM uses an implicit high-dimensional feature space",
            "Yes, RBF is always exactly 3D",
            "No, SVM has no kernel function",
        ],
        "answer": "No, true RBF SVM uses an implicit high-dimensional feature space",
    },
]

score = 0
for i, item in enumerate(questions, start=1):
    choice = st.radio(f"{i}. {item['q']}", item["options"], key=f"quiz_{i}")
    if choice == item["answer"]:
        score += 1

if st.button("Check score"):
    st.success(f"Score: {score} / {len(questions)}")
    if score == len(questions):
        st.balloons()
