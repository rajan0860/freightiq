"""
Natural Language Query page — ask questions about the supply chain knowledge base.
"""

from __future__ import annotations

import streamlit as st
import requests

API_BASE = "http://localhost:8000"


def render():
    st.header("💬 Ask FreightIQ")
    st.caption("Query the supply chain knowledge base in natural language.")

    # Query input
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. Which Asia-Europe routes are most at risk this week?",
    )

    if st.button("🔍 Search", use_container_width=False, disabled=not question):
        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()

                    st.markdown("### Answer")
                    st.markdown(data.get("answer", "No answer returned."))

                    sources = data.get("sources", [])
                    if sources:
                        st.markdown("---")
                        st.markdown(f"**Sources:** {', '.join(sources)}")
                else:
                    st.error(f"API error: {resp.status_code} — {resp.text}")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the FastAPI backend running?")

    # Example prompts
    st.divider()
    st.markdown("#### 💡 Example Questions")
    examples = [
        "Port of Rotterdam strike — which shipments are affected?",
        "What historical disruptions have affected the Shanghai to Rotterdam route?",
        "Which carriers have the highest reliability scores?",
        "What is the weather risk at major Asian ports?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex
            st.rerun()


if __name__ == "__main__":
    render()
