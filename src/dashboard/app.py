"""
FreightIQ Streamlit Dashboard — main entry point.

Run with:
    streamlit run src/dashboard/app.py

Requires the FastAPI backend running on port 8000:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FreightIQ — Supply Chain Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import page renderers
# ---------------------------------------------------------------------------
from src.dashboard.components import render_header
from src.dashboard.pages import alerts as alerts_page
from src.dashboard.pages import risk_table as risk_table_page
from src.dashboard.pages import region_map as region_map_page
from src.dashboard.pages import query as query_page

# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------
render_header()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("# 🚢 FreightIQ")
st.sidebar.caption("AI-Powered Supply Chain Intelligence")
st.sidebar.divider()

PAGES = {
    "🚨 Live Alerts": alerts_page,
    "📊 Risk Table": risk_table_page,
    "🗺️ Region Map": region_map_page,
    "💬 Ask FreightIQ": query_page,
}

selection = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Quick Start**
    1. Start the API: `uvicorn src.api.main:app --reload`
    2. Click **Run Agent** on the Alerts page
    3. Explore risk scores and ask questions

    ---
    *Built with LangGraph, XGBoost, FastAPI & Streamlit*
    """
)

# ---------------------------------------------------------------------------
# Render selected page
# ---------------------------------------------------------------------------
PAGES[selection].render()
