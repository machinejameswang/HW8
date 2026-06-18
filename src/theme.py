"""Shared visual constants for the SVM dashboard."""

COLORS = {
    "background": "#090B10",
    "panel": "rgba(255, 255, 255, 0.075)",
    "panel_border": "rgba(255, 255, 255, 0.16)",
    "text": "#F8FAFC",
    "muted": "#AAB4C5",
    "accent": "#6366F1",
    "cyan": "#22D3EE",
    "amber": "#F59E0B",
    "class_inner": "#38BDF8",
    "class_outer": "#F43F5E",
    "margin": "#A78BFA",
    "boundary": "#F8FAFC",
}

PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": COLORS["text"], "family": "Inter, Noto Sans TC, sans-serif"},
    "margin": {"l": 24, "r": 24, "t": 54, "b": 28},
}


def streamlit_css() -> str:
    """Return custom CSS for the dark glassmorphism Streamlit interface."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: Inter, "Noto Sans TC", system-ui, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 18% 12%, rgba(99, 102, 241, 0.16), transparent 28%),
            radial-gradient(circle at 82% 8%, rgba(34, 211, 238, 0.12), transparent 24%),
            linear-gradient(135deg, #090B10 0%, #0D1422 46%, #090B10 100%);
        color: {COLORS["text"]};
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(11, 17, 29, 0.82);
        border-right: 1px solid {COLORS["panel_border"]};
        backdrop-filter: blur(20px);
    }}

    div[data-testid="stMetric"], .glass-panel {{
        background: {COLORS["panel"]};
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(18px);
    }}

    div[data-testid="stMetric"] label {{
        color: {COLORS["muted"]};
        font-size: 0.82rem;
    }}

    h1, h2, h3 {{
        letter-spacing: 0;
    }}

    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}

    .hero-subtitle {{
        color: {COLORS["muted"]};
        margin-bottom: 1rem;
        max-width: 980px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.45rem 0.8rem;
    }}
    </style>
    """
