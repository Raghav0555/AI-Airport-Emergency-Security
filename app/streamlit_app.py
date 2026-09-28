import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from simulation.orchestrator import run_simulation
from src.world.graph_loader import load_airport_graph, node_positions
from src.world.scenario_loader import build_initial_state, load_scenarios

st.set_page_config(page_title="AI Airport Emergency & Security Simulator", layout="wide")

NODE_COLORS = {
    "corridor": "#B0B0B0",
    "checkpoint": "#4C72B0",
    "terminal": "#2E7D32",
    "gate": "#66BB6A",
    "emergency_station": "#EF6C00",
    "restricted": "#8E24AA",
    "staff_area": "#AB47BC",
    "exit": "#0288D1",
}


@st.cache_resource
def get_graph():
    return load_airport_graph()


def draw_map(graph, state):
    pos = node_positions(graph)
    fig, ax = plt.subplots(figsize=(11, 6))

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#CCCCCC", width=1.2)

    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, d in graph.nodes(data=True) if d["type"] == node_type]
        if nodes:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=nodes, node_color=color,
                node_size=260, ax=ax, label=node_type,
            )

    blocked = [n for n in state.blocked_nodes if n in graph.nodes]
    if blocked:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=blocked, node_color="#D32F2F",
            node_shape="X", node_size=340, ax=ax, label="blocked",
        )

    if state.threat_location and state.threat_location in graph.nodes:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=[state.threat_location], node_color="#B71C1C",
            node_shape="*", node_size=600, ax=ax, label="threat",
        )

    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=8)
    ax.set_title("Airport Layout")
    ax.axis("off")
    return fig


def main():
    st.title("AI Airport Emergency & Security Management Simulator")

    graph = get_graph()
    scenarios = load_scenarios()

    st.sidebar.header("Scenario")
    scenario_name = st.sidebar.selectbox(
        "Select a scenario",
        options=list(scenarios.keys()),
        format_func=lambda k: scenarios[k]["label"],
    )
    st.sidebar.markdown(scenarios[scenario_name]["description"])

    state = build_initial_state(graph, scenario_name)
    result = run_simulation(graph, state, scenario_name)

    col_map, col_side = st.columns([2, 1])

    with col_map:
        st.subheader("Airport Map")
        st.pyplot(draw_map(graph, state))

    with col_side:
        st.subheader("Active Incidents")
        if state.active_incidents:
            for inc in state.active_incidents:
                st.markdown(
                    f"- **{inc.type}** at `{inc.location_node}` "
                    f"(severity: {inc.severity})"
                )
        else:
            st.markdown("No active incidents - normal operation.")

        st.subheader("World State")
        st.markdown(f"- Blocked nodes: `{sorted(state.blocked_nodes) or 'none'}`")
        st.markdown(f"- Threat location: `{state.threat_location or 'none'}`")

        st.subheader("AI Decision Log")
        for line in result.decision_log:
            st.markdown(f"- {line}")

    with st.expander("Raw event log"):
        for line in state.event_log:
            st.text(line)


main()