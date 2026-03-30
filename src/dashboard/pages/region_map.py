"""
Region Map page — geographic view of disruption risk by region.
"""

from __future__ import annotations

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8000"

# Approximate center coordinates for each region
REGION_COORDS = {
    "Europe": {"lat": 50.0, "lon": 10.0},
    "North America": {"lat": 40.0, "lon": -100.0},
    "Asia": {"lat": 35.0, "lon": 105.0},
    "South America": {"lat": -15.0, "lon": -55.0},
    "Middle East": {"lat": 25.0, "lon": 45.0},
    "Africa": {"lat": 5.0, "lon": 20.0},
}


def render():
    st.header("🗺️ Disruption Risk Map")
    st.caption("Risk concentration by region, based on shipment scores.")

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
        st.info("No shipment data available.")
        return

    df = pd.DataFrame(data)

    # Aggregate by region
    region_agg = df.groupby("region").agg(
        avg_risk=("risk_score", "mean"),
        shipment_count=("shipment_id", "count"),
        high_risk_count=("risk_level", lambda x: (x == "HIGH").sum()),
    ).reset_index()

    # Map coordinates
    region_agg["lat"] = region_agg["region"].map(lambda r: REGION_COORDS.get(r, {}).get("lat", 0))
    region_agg["lon"] = region_agg["region"].map(lambda r: REGION_COORDS.get(r, {}).get("lon", 0))

    # Plotly scatter map
    fig = px.scatter_geo(
        region_agg,
        lat="lat",
        lon="lon",
        size="shipment_count",
        color="avg_risk",
        hover_name="region",
        hover_data={
            "avg_risk": ":.2f",
            "shipment_count": True,
            "high_risk_count": True,
            "lat": False,
            "lon": False,
        },
        color_continuous_scale="RdYlGn_r",
        size_max=40,
        title="",
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Region Summary")
    st.dataframe(
        region_agg[["region", "shipment_count", "avg_risk", "high_risk_count"]].rename(
            columns={
                "region": "Region",
                "shipment_count": "Shipments",
                "avg_risk": "Avg Risk",
                "high_risk_count": "High Risk",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    render()
