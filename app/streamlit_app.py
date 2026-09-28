import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation.orchestrator import run_simulation
from src.world.graph_loader import load_airport_graph, node_positions
from src.world.scenario_loader import build_initial_state, load_scenarios

st.set_page_config(page_title="AI Airport Emergency & Security Simulator", layout="wide")
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background-color: #0A0F1C; }
section[data-testid="stSidebar"] { background-color: #0D1424; border-right: 1px solid #1F2937; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
h1 { font-weight: 700; color: #F1F5F9; }

.mono { font-family: 'JetBrains Mono', monospace; }

.status-strip {
    display: flex; align-items: center; gap: 28px;
    padding: 14px 20px; margin: 6px 0 22px 0;
    background: #111826; border: 1px solid #1F2937; border-radius: 10px;
}
.status-badge {
    display: flex; align-items: center; gap: 9px;
    padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
}
.status-dot { width: 9px; height: 9px; border-radius: 50%; }
.status-nominal { background: rgba(52,211,153,0.12); color: #34D399; }
.status-nominal .status-dot { background: #34D399; box-shadow: 0 0 8px #34D399; }
.status-monitoring { background: rgba(251,191,36,0.12); color: #FBBF24; }
.status-monitoring .status-dot { background: #FBBF24; box-shadow: 0 0 8px #FBBF24; }
.status-active { background: rgba(248,113,113,0.14); color: #F87171; }
.status-active .status-dot { background: #F87171; box-shadow: 0 0 8px #F87171; }

.kpi { flex: 1; }
.kpi-label { font-size: 0.78rem; color: #7C8AA3; margin-bottom: 2px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #E5E9F0; }
.kpi-divider { width: 1px; height: 34px; background: #1F2937; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background-color: #111826; border: 1px solid #1F2937; border-radius: 8px 8px 0 0;
    color: #7C8AA3; font-family: 'Space Grotesk', sans-serif; font-weight: 500;
}
.stTabs [aria-selected="true"] { color: #38BDF8 !important; border-bottom: 2px solid #38BDF8 !important; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

BG = "#0B1220"
TEXT = "#CBD5E1"

DISPLAY_OVERRIDES = {"STAFF_2": (950, 70)}

NODE_STYLE = {
    "corridor": dict(name="Corridor", symbol="circle", size=13, color="#94A3B8", label_pos="top right"),
    "checkpoint": dict(name="Checkpoint", symbol="diamond", size=17, color="#3B82F6", label_pos="middle right"),
    "terminal": dict(name="Terminal", symbol="square", size=24, color="#22C55E", label_pos="bottom right"),
    "gate": dict(name="Gate", symbol="circle", size=15, color="#86EFAC", label_pos="top center"),
    "emergency_station": dict(name="Emergency station", symbol="hexagon", size=20, color="#FB923C", label_pos="bottom center"),
    "restricted": dict(name="Restricted zone", symbol="octagon", size=20, color="#A855F7", label_pos="top center"),
    "staff_area": dict(name="Staff area", symbol="pentagon", size=18, color="#C084FC", label_pos="top center"),
    "exit": dict(name="Exit", symbol="triangle-down", size=20, color="#38BDF8", label_pos="bottom center"),
}

AGENT_STYLE = {
    "security": dict(name="Security officer", symbol="square", color="#3B82F6", size=11, offset=(-16, 22)),
    "emergency": dict(name="Emergency team", symbol="triangle-up", color="#EF4444", size=12, offset=(16, 22)),
    "staff": dict(name="Staff", symbol="diamond", color="#E879F9", size=11, offset=(-16, -22)),
    "passenger": dict(name="Passenger", symbol="circle", color="#F8FAFC", size=9, offset=(16, -22)),
}

TERMINAL_CENTERS = {"TERMINAL 1": 200, "TERMINAL 2": 500, "TERMINAL 3": 800}


@st.cache_resource
def get_graph():
    return load_airport_graph()


def short_label(node_id, node_type):
    number = node_id.split("_")[-1]
    return {
        "corridor": f"C{number}",
        "checkpoint": f"CP{number}",
        "terminal": f"T{number}",
        "gate": number,
        "emergency_station": f"EMS {number}",
        "restricted": f"Restricted {number}",
        "staff_area": f"Staff {number}",
        "exit": f"Exit {number}",
    }[node_type]


def edge_trace(xs, ys, color, width, dash="solid"):
    return go.Scatter(
        x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color=color, width=width, dash=dash),
    )


def build_map(graph, state, path):
    pos = node_positions(graph)
    for node, xy in DISPLAY_OVERRIDES.items():
        if node in pos:
            pos[node] = xy

    fig = go.Figure()

    groups = {"main": ([], []), "backbone": ([], []), "restricted": ([], []), "blocked": ([], [])}
    for u, v in graph.edges:
        types = (graph.nodes[u]["type"], graph.nodes[v]["type"])
        if u in state.blocked_nodes or v in state.blocked_nodes:
            key = "blocked"
        elif types == ("corridor", "corridor"):
            key = "backbone"
        elif any(t in ("restricted", "staff_area") for t in types):
            key = "restricted"
        else:
            key = "main"
        groups[key][0].extend([pos[u][0], pos[v][0], None])
        groups[key][1].extend([pos[u][1], pos[v][1], None])

    fig.add_trace(edge_trace(*groups["backbone"], "#334155", 7))
    fig.add_trace(edge_trace(*groups["main"], "#475569", 2))
    fig.add_trace(edge_trace(*groups["restricted"], "#7C3AED", 2, "dot"))
    fig.add_trace(edge_trace(*groups["blocked"], "#EF4444", 3, "dash"))

    if path and len(path) > 1:
        px = [pos[n][0] for n in path]
        py = [pos[n][1] for n in path]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="rgba(251,191,36,0.28)", width=16)))
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", name="Selected route", hoverinfo="skip",
                                 line=dict(color="#FBBF24", width=4),
                                 marker=dict(size=6, color="#FBBF24")))

    agents_at = {}
    for kind in AGENT_STYLE:
        for agent_id, node in getattr(state, f"{kind}_locations").items():
            agents_at.setdefault(node, []).append(f"{agent_id} ({kind})")

    for node_type, style in NODE_STYLE.items():
        nodes = [n for n, d in graph.nodes(data=True) if d["type"] == node_type]
        if not nodes:
            continue
        hover = []
        for n in nodes:
            data = graph.nodes[n]
            lines = [f"<b>{n}</b>", f"type: {node_type}", f"capacity: {data['capacity']}"]
            if data["zone_restricted"]:
                lines.append("restricted access")
            if n in state.blocked_nodes:
                lines.append("<b>BLOCKED</b>")
            if n == state.threat_location:
                lines.append("<b>THREAT</b>")
            if n in agents_at:
                lines.append("agents: " + ", ".join(agents_at[n]))
            hover.append("<br>".join(lines))

        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            name=style["name"],
            text=[short_label(n, node_type) for n in nodes],
            textposition=style["label_pos"],
            textfont=dict(size=10 if node_type != "gate" else 9, color=TEXT),
            hovertext=hover,
            hoverinfo="text",
            marker=dict(symbol=style["symbol"], size=style["size"], color=style["color"],
                        line=dict(color=BG, width=2)),
        ))

    blocked = [n for n in state.blocked_nodes if n in pos]
    if blocked:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in blocked], y=[pos[n][1] for n in blocked],
            mode="markers", name="Blocked", hoverinfo="skip",
            marker=dict(symbol="x", size=26, color="#EF4444", line=dict(color="#EF4444", width=4)),
        ))

    if state.threat_location and state.threat_location in pos:
        tx, ty = pos[state.threat_location]
        fig.add_trace(go.Scatter(
            x=[tx], y=[ty], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(symbol="circle", size=52, color="rgba(239,68,68,0.22)",
                        line=dict(color="rgba(239,68,68,0.6)", width=1)),
        ))
        fig.add_trace(go.Scatter(
            x=[tx], y=[ty], mode="markers", name="Threat", hoverinfo="skip",
            marker=dict(symbol="star", size=26, color="#EF4444", line=dict(color="#FFFFFF", width=1.5)),
        ))

    for kind, style in AGENT_STYLE.items():
        locations = getattr(state, f"{kind}_locations")
        if not locations:
            continue
        dx, dy = style["offset"]
        fig.add_trace(go.Scatter(
            x=[pos[node][0] + dx for node in locations.values()],
            y=[pos[node][1] + dy for node in locations.values()],
            mode="markers", name=style["name"],
            hovertext=[f"<b>{a}</b><br>{kind}<br>at {n}" for a, n in locations.items()],
            hoverinfo="text",
            marker=dict(symbol=style["symbol"], size=style["size"], color=style["color"],
                        line=dict(color=BG, width=1.5)),
        ))

    if path and len(path) > 1:
        fig.add_trace(go.Scatter(
            x=[pos[path[0]][0]], y=[pos[path[0]][1]], mode="markers", name="Route start", hoverinfo="skip",
            marker=dict(symbol="circle-open", size=26, color="#22C55E", line=dict(color="#22C55E", width=3)),
        ))
        fig.add_trace(go.Scatter(
            x=[pos[path[-1]][0]], y=[pos[path[-1]][1]], mode="markers", name="Route end", hoverinfo="skip",
            marker=dict(symbol="star-open", size=28, color="#FBBF24", line=dict(color="#FBBF24", width=3)),
        ))

    shapes, annotations = [], []
    for title, cx in TERMINAL_CENTERS.items():
        shapes.append(dict(
            type="rect", x0=cx - 120, x1=cx + 120, y0=100, y1=252, layer="below",
            fillcolor="rgba(34,197,94,0.06)", line=dict(color="rgba(34,197,94,0.35)", width=1, dash="dot"),
        ))
        annotations.append(dict(
            x=cx, y=268, text=f"<b>{title}</b>", showarrow=False,
            font=dict(size=11, color="#86EFAC"),
        ))

    fig.update_layout(
        height=650,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=70),
        xaxis=dict(visible=False, range=[30, 1070]),
        yaxis=dict(visible=False, range=[-200, 290]),
        legend=dict(orientation="h", y=-0.02, yanchor="top", x=0.5, xanchor="center",
                    font=dict(color=TEXT, size=11)),
        hoverlabel=dict(bgcolor="#1E293B", font_color="#F8FAFC"),
        dragmode="pan",
        shapes=shapes,
        annotations=annotations,
    )
    return fig


def ga_chart(ga):
    generations = list(range(1, len(ga["best_history"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=generations, y=ga["best_history"], name="best fitness",
                             line=dict(color="#22C55E", width=3)))
    fig.add_trace(go.Scatter(x=generations, y=ga["avg_history"], name="average fitness",
                             line=dict(color="#F59E0B", width=2)))
    fig.update_layout(
        template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="generation", yaxis_title="fitness (log scale, lower is better)",
        yaxis_type="log", legend=dict(orientation="h", y=1.12),
    )
    return fig


def route_label(row):
    return f"{row['agent']} | {row['purpose']} | {row['start']} -> {row['destination']}"


def module_summary(result):
    csp = result.csp_report
    ga = result.ga_assignment
    mm = result.minimax_outcome

    if result.expert_system_trace:
        expert = "; ".join(f"{f['rule']} -> {f['action']}" for f in result.expert_system_trace)
    else:
        expert = "No rules fired"

    if csp["violations"]:
        csp_text = f"{len(csp['violations'])} violation(s) found"
    else:
        csp_text = (
            f"Valid: {csp['routes_checked']} routes and "
            f"{csp['assignments_checked']} assignments checked, 0 violations"
        )
    if csp["access_denied"]:
        denied = sum(1 for a in csp["access_denied"] if not a["allowed"])
        csp_text += f"; {denied} access attempt(s) denied"

    rows = [
        ("Expert System", expert),
        ("CSP", csp_text),
        ("A*", f"{len(result.route_table)} route(s) calculated"),
        ("Genetic Algorithm", f"Optimized assignment, best fitness {ga['best_fitness']}" if ga else "Not triggered"),
        ("Minimax", f"Threat {mm['outcome']} in {mm['rounds']} round(s)" if mm else "Not triggered"),
        ("Face Recognition", "Not connected yet"),
    ]
    return pd.DataFrame(rows, columns=["Module", "Decision"])
def system_status(state, result):
    if not state.active_incidents:
        return "nominal", "Nominal", "All systems normal"
    severities = [i.severity for i in state.active_incidents]
    if "high" in severities:
        return "active", "Active Response", f"{len(state.active_incidents)} incident(s) in progress"
    return "monitoring", "Monitoring", f"{len(state.active_incidents)} incident(s) tracked"


def render_status_strip(state, result):
    level, label, detail = system_status(state, result)
    responders = len(result.ga_assignment["rows"]) if result.ga_assignment else 0
    threat = result.minimax_outcome["outcome"] if result.minimax_outcome else "none"
    html = f'''
    <div class="status-strip">
        <div class="status-badge status-{level}"><div class="status-dot"></div>{label}</div>
        <div class="mono" style="color:#7C8AA3; font-size:0.85rem;">{detail}</div>
        <div style="flex:1"></div>
        <div class="kpi"><div class="kpi-label">Active incidents</div><div class="kpi-value">{len(state.active_incidents)}</div></div>
        <div class="kpi-divider"></div>
        <div class="kpi"><div class="kpi-label">Blocked nodes</div><div class="kpi-value">{len(state.blocked_nodes)}</div></div>
        <div class="kpi-divider"></div>
        <div class="kpi"><div class="kpi-label">Responders deployed</div><div class="kpi-value">{responders}</div></div>
        <div class="kpi-divider"></div>
        <div class="kpi"><div class="kpi-label">Threat status</div><div class="kpi-value">{threat}</div></div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

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
    state = build_initial_state(graph, scenario_name)
    result = run_simulation(graph, state, scenario_name)

    render_status_strip(state, result)

    col_map, col_side = st.columns([2.2, 1])
    col_map, col_side = st.columns([2.2, 1])

    with col_map:
        st.subheader("Airport Map")
        drawable = [r for r in result.route_table if r["path"]]
        choice = st.selectbox(
            "Show a route on the map",
            options=["None"] + [route_label(r) for r in drawable],
        )
        selected = None
        for r in drawable:
            if route_label(r) == choice:
                selected = r
        st.plotly_chart(
            build_map(graph, state, selected["path"] if selected else None),
            width="stretch",
            config={"displayModeBar": False},
        )
        if selected:
            old = f", was {selected['old_distance']}" if selected["old_distance"] is not None else ""
            st.caption(f"Distance {selected['distance']}{old} | status: {selected['status']}")

    with col_side:
        st.subheader("Active Incidents")
        if state.active_incidents:
            for inc in state.active_incidents:
                box = {"high": st.error, "medium": st.warning}.get(inc.severity, st.info)
                box(f"**{inc.type}** at `{inc.location_node}`  \nseverity: {inc.severity}")
        else:
            st.success("No active incidents - normal operation.")

        st.subheader("World State")
        st.markdown(f"- Blocked nodes: `{sorted(state.blocked_nodes) or 'none'}`")
        st.markdown(f"- Threat location: `{state.threat_location or 'none'}`")

        st.subheader("AI Decision Log")
        for line in result.decision_log:
            st.markdown(f"- {line}")

    tab_ai, tab_routes, tab_csp, tab_ga, tab_threat, tab_id = st.tabs(
        ["AI Decisions", "Routes", "Constraints", "Optimization (GA)", "Threat Simulation", "Identity"]
    )

    with tab_ai:
        st.table(module_summary(result))
        for fired in result.expert_system_trace:
            st.markdown(f"- **{fired['rule']}**: {fired['explanation']}")

    with tab_routes:
        if result.route_table:
            table = pd.DataFrame(
                [
                    {
                        "agent": r["agent"],
                        "purpose": r["purpose"],
                        "start": r["start"],
                        "destination": r["destination"],
                        "distance": r["distance"],
                        "old distance": r["old_distance"],
                        "status": r["status"],
                        "path": " -> ".join(r["path"]),
                    }
                    for r in result.route_table
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("No routes calculated for this scenario.")

    with tab_csp:
        csp = result.csp_report
        c1, c2, c3 = st.columns(3)
        c1.metric("Routes checked", csp["routes_checked"])
        c2.metric("Assignments checked", csp["assignments_checked"])
        c3.metric("Violations", len(csp["violations"]))
        if csp["violations"]:
            for v in csp["violations"]:
                st.error(v)
        else:
            st.success("All constraints passed for the planned routes and assignments.")
        for attempt in csp["access_denied"]:
            if attempt["allowed"]:
                st.success(f"{attempt['agent']} -> {attempt['target']}: access allowed")
            else:
                st.warning(f"{attempt['agent']} -> {attempt['target']}: " + "; ".join(attempt["reasons"]))
        st.markdown(f"Blocked nodes: `{sorted(state.blocked_nodes) or 'none'}`")
        st.markdown(f"Restricted zones: `{sorted(state.restricted_zones)}`")

    with tab_ga:
        ga = result.ga_assignment
        if ga:
            c1, c2, c3 = st.columns(3)
            c1.metric("Best fitness", ga["best_fitness"])
            c2.metric("Generations", ga["generations"])
            c3.metric("A* calls", ga["astar_calls"])
            st.dataframe(pd.DataFrame(ga["rows"]), width="stretch", hide_index=True)
            st.plotly_chart(ga_chart(ga), width="stretch", config={"displayModeBar": False})
        else:
            st.info("The genetic algorithm runs when there are incidents that need responders.")

    with tab_threat:
        mm = result.minimax_outcome
        if mm:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Outcome", mm["outcome"])
            c2.metric("Rounds", mm["rounds"])
            c3.metric("Peak risk", mm["peak_risk"])
            c4.metric("Nodes evaluated", mm["nodes_evaluated"])
            st.dataframe(pd.DataFrame(mm["log"]), width="stretch", hide_index=True)
            base = mm["chase_baseline"]
            st.caption(
                f"Baseline without minimax (simple chase): {base['outcome']} in "
                f"{base['rounds']} round(s), peak risk {base['peak_risk']}"
            )
        else:
            st.info("No security threat in this scenario.")

    with tab_id:
        st.info("Face recognition module is not connected yet.")

    with st.expander("Raw event log"):
        for line in state.event_log:
            st.text(line)


main()