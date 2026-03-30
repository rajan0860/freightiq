"""
Reusable UI components for the FreightIQ Streamlit dashboard.
"""

from __future__ import annotations

import streamlit as st


def render_header():
    """Render the common page header."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .main { font-family: 'Inter', sans-serif; }
        .header-title { font-size: 2rem; font-weight: 700; }
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 12px;
            padding: 1.2rem;
            color: white;
            text-align: center;
        }
        .metric-value { font-size: 1.8rem; font-weight: 700; }
        .metric-label { font-size: 0.85rem; color: #94a3b8; }
        .severity-high { color: #ef4444; font-weight: 700; }
        .severity-medium { color: #f59e0b; font-weight: 700; }
        .severity-low { color: #22c55e; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(level: str) -> str:
    """Return an HTML badge for severities."""
    colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
    color = colors.get(level, "#64748b")
    return (
        f'<span style="background:{color}; color:white; padding:2px 10px; '
        f'border-radius:12px; font-size:0.8rem; font-weight:600;">{level}</span>'
    )


def metric_card(label: str, value: str | int | float):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
