"""
Risk Table page — shipments ranked by disruption risk.
"""

from __future__ import annotations

import streamlit as st
import requests
import pandas as pd

from src.dashboard.components import severity_badge

API_BASE = "http://localhost:8000"


def render():
    st.header("📊 Shipment Risk Table")
    st.caption("All shipments ranked by XGBoost risk score (highest risk first).")

    try:
        resp = requests.get(f"{API_BASE}/shipments/risk", timeout=30)
        if resp.ok:
            data = resp.json()
        else:
            st.error(f"API returned {resp.status_code}")
            return
    except requests.ConnectionError:
        st.warning("⚠️ Cannot connect to API. Run: `uvicorn src.api.main:app --reload --port 8000`")
        return

    if not data:
        st.info("No shipment data. Run `python scripts/generate_data.py` first.")
        return

    df = pd.DataFrame(data)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    high_count = len(df[df["risk_level"] == "HIGH"])
    med_count = len(df[df["risk_level"] == "MEDIUM"])
    low_count = len(df[df["risk_level"] == "LOW"])
    avg_score = df["risk_score"].mean()

    col1.metric("🔴 High Risk", high_count)
    col2.metric("🟡 Medium Risk", med_count)
    col3.metric("🟢 Low Risk", low_count)
    col4.metric("Avg Risk Score", f"{avg_score:.2f}")

    st.divider()

    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )
    with filter_col2:
        route_filter = st.multiselect(
            "Filter by Route",
            options=sorted(df["route"].unique().tolist()),
        )

    filtered = df[df["risk_level"].isin(risk_filter)]
    if route_filter:
        filtered = filtered[filtered["route"].isin(route_filter)]

    # Display table
    st.dataframe(
        filtered[["shipment_id", "route", "carrier", "region", "risk_score", "risk_level", "explanation"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
        },
    )


if __name__ == "__main__":
    render()
