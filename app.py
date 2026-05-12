import streamlit as st

from forms.adherence_form import render_adherence_form
from forms.injury_form import render_injury_form
from dashboards.dashboard import render_dashboard

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Gym Smart AI",
    layout="wide"
)

# -----------------------------
# PREMIUM UI STYLE
# -----------------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    .stNumberInput, .stSelectbox {
        margin-bottom: 12px;
    }

    button[kind="primary"] {
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 AI Gym Plataforma de Predicción")

# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.selectbox(
    "Modulo de Selección",
    [
        "Dashboard",
        "Predicción de Adherencia",
        "Predicción de Riesgo de Lesión"
    ]
)

# -----------------------------
# ROUTING
# -----------------------------
if menu == "Predicción de Adherencia":
    render_adherence_form()

elif menu == "Predicción de Riesgo de Lesión":
    render_injury_form()

elif menu == "Dashboard":
    render_dashboard()