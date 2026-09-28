from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class Incident:
    incident_id: str
    type: str
    location_node: str
    severity: str


@dataclass(frozen=True)
class WorldState:
    passenger_locations: Dict[str, str] = field(default_factory=dict)
    security_locations: Dict[str, str] = field(default_factory=dict)
    emergency_locations: Dict[str, str] = field(default_factory=dict)
    staff_locations: Dict[str, str] = field(default_factory=dict)

    threat_location: Optional[str] = None
    blocked_nodes: Set[str] = field(default_factory=set)
    active_gates: Set[str] = field(default_factory=set)
    restricted_zones: Set[str] = field(default_factory=set)
    occupancy: Dict[str, int] = field(default_factory=dict)
    permissions: Dict[str, str] = field(default_factory=dict)
    active_incidents: List[Incident] = field(default_factory=list)
    event_log: List[str] = field(default_factory=list)

    def with_event(self, message: str) -> "WorldState":
        return replace(self, event_log=self.event_log + [message])

    def block_node(self, node_id: str, reason: str = "") -> "WorldState":
        new_blocked = set(self.blocked_nodes) | {node_id}
        msg = f"Blocked node {node_id}" + (f" ({reason})" if reason else "")
        return replace(self, blocked_nodes=new_blocked).with_event(msg)

    def unblock_node(self, node_id: str) -> "WorldState":
        new_blocked = set(self.blocked_nodes) - {node_id}
        return replace(self, blocked_nodes=new_blocked).with_event(
            f"Unblocked node {node_id}"
        )

    def move_agent(self, agent_id: str, agent_type: str, node_id: str) -> "WorldState":
        field_name = f"{agent_type}_locations"
        current = dict(getattr(self, field_name))
        current[agent_id] = node_id
        msg = f"{agent_type}:{agent_id} moved to {node_id}"
        return replace(self, **{field_name: current}).with_event(msg)

    def deploy_agent(self, agent_id: str, agent_type: str, node_id: str) -> "WorldState":
        return self.move_agent(agent_id, agent_type, node_id).with_event(
            f"Deployed {agent_type}:{agent_id} to {node_id}"
        )

    def set_threat_location(self, node_id: Optional[str]) -> "WorldState":
        return replace(self, threat_location=node_id).with_event(
            f"Threat location set to {node_id}"
        )

    def add_incident(self, incident: Incident) -> "WorldState":
        return replace(
            self, active_incidents=self.active_incidents + [incident]
        ).with_event(
            f"Incident {incident.incident_id} ({incident.type}) reported at {incident.location_node}"
        )

    def resolve_incident(self, incident_id: str) -> "WorldState":
        remaining = [i for i in self.active_incidents if i.incident_id != incident_id]
        return replace(self, active_incidents=remaining).with_event(
            f"Incident {incident_id} resolved"
        )

    def evacuate_gate(self, gate_id: str) -> "WorldState":
        new_active = set(self.active_gates) - {gate_id}
        return replace(self, active_gates=new_active).with_event(
            f"Gate {gate_id} evacuated / closed"
        )

    def open_exit(self, exit_id: str) -> "WorldState":
        new_blocked = set(self.blocked_nodes) - {exit_id}
        return replace(self, blocked_nodes=new_blocked).with_event(
            f"Exit {exit_id} confirmed open"
        )

    def close_restricted_zone(self, zone_id: str) -> "WorldState":
        new_restricted = set(self.restricted_zones) | {zone_id}
        new_blocked = set(self.blocked_nodes) | {zone_id}
        return replace(
            self, restricted_zones=new_restricted, blocked_nodes=new_blocked
        ).with_event(f"Restricted zone {zone_id} locked down")

    def deep_copy(self) -> "WorldState":
        return copy.deepcopy(self)


if __name__ == "__main__":
    s0 = WorldState()
    s1 = s0.block_node("COR_5", reason="test")
    s2 = s1.move_agent("SEC_1", "security", "COR_3")
    print("s0 blocked:", s0.blocked_nodes)
    print("s1 blocked:", s1.blocked_nodes)
    print("s2 security:", s2.security_locations)
    print("s0 unchanged:", s0.blocked_nodes == set() and s0.security_locations == {})
    print("log:", s2.event_log)