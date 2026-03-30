"""
LangGraph agent definition for FreightIQ.

Defines a linear graph: Detect → Retrieve → Score → Recommend

Usage:
    from src.agent.graph import DisruptionAgent

    agent = DisruptionAgent()
    alerts = agent.run()
"""

from __future__ import annotations

from typing import List

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState, Alert
from src.agent.nodes import detect_node, retrieve_node, score_node, recommend_node
from src.agent.alerts import sort_alerts_by_severity, format_alerts_summary


def build_graph() -> StateGraph:
    """Construct the LangGraph state graph."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect", detect_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("score", score_node)
    graph.add_node("recommend", recommend_node)

    # Define edges: linear pipeline
    graph.set_entry_point("detect")
    graph.add_edge("detect", "retrieve")
    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "recommend")
    graph.add_edge("recommend", END)

    return graph


class DisruptionAgent:
    """
    High-level wrapper around the LangGraph disruption detection agent.

    Usage:
        agent = DisruptionAgent()
        alerts = agent.run(raw_feeds=news_items, shipments=shipment_records)
    """

    def __init__(self):
        graph = build_graph()
        self.app = graph.compile()

    def run(
        self,
        raw_feeds: List[dict] | None = None,
        shipments: List[dict] | None = None,
    ) -> List[Alert]:
        """
        Execute the full agent pipeline.

        Parameters
        ----------
        raw_feeds : list[dict]
            News + weather items from the ingestion pipeline.
        shipments : list[dict]
            Current shipment records (with feature columns) to score.

        Returns
        -------
        list[Alert]
            Sorted list of disruption alerts (most severe first).
        """
        initial_state = {
            "raw_feeds": raw_feeds or [],
            "shipments": shipments or [],
            "events": [],
            "context": "",
            "scored_shipments": [],
            "alerts": [],
            "error": None,
        }

        result = self.app.invoke(initial_state)

        alerts = result.get("alerts", [])
        return sort_alerts_by_severity(alerts)

    def run_and_print(
        self,
        raw_feeds: List[dict] | None = None,
        shipments: List[dict] | None = None,
    ) -> List[Alert]:
        """Run the agent and print a human-readable summary."""
        alerts = self.run(raw_feeds=raw_feeds, shipments=shipments)
        print(format_alerts_summary(alerts))
        return alerts
