"""
Natural Language Query page — ask questions about the supply chain knowledge base.
"""

from __future__ import annotations

import streamlit as st
import os
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def set_query(query_text: str):
    """Callback to update query input and trigger search."""
    st.session_state["query_input_box"] = query_text
    st.session_state["trigger_search"] = True


def render():
    st.header("💬 Ask FreightIQ")
    st.caption("Query the supply chain knowledge base in natural language.")

    # Initialize session state
    if "query_input_box" not in st.session_state:
        st.session_state["query_input_box"] = ""
    if "trigger_search" not in st.session_state:
        st.session_state["trigger_search"] = False

    # Query input
    question = st.text_input(
        "Ask a question",
        key="query_input_box",
        placeholder="e.g. Which Asia-Europe routes are most at risk this week?",
    )

    # Trigger search if button clicked OR auto-trigger flag is set
    search_clicked = st.button("🔍 Search", width="content", disabled=not question)
    
    if search_clicked or st.session_state["trigger_search"]:
        # Reset the trigger flag
        st.session_state["trigger_search"] = False
        
        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question},
                    timeout=180,  # 3 minutes for NL query
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

            except requests.Timeout:
                st.warning("⏱️ The search is taking longer than expected. Please try again or simplify your question.")
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
        st.button(ex, key=f"btn_{ex}", on_click=set_query, args=(ex,))


if __name__ == "__main__":
    render()
