from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime, timezone


# Action names align 1:1 with env action_space (env/spaces.py).
# Keeping them here as a single source of truth for swarm prompts and
# resonance aggregation. Order matters — index = env action index.
ACTION_NAMES: list[str] = [
    "funding_boost",
    "teacher_incentive",
    "student_scholarship",
    "attendance_mandate",
    "resource_realloc",
    "transparency_report",
    "staff_hiring",
    "counseling_programs",
]


@dataclass
class Verdict:
    """One agent's recommendation."""
    persona_id: str
    persona_name: str
    role: str
    action_vector: list[float]   # length 8, in [0, 1]
    reasoning: str
    confidence: float            # in [0, 1]
    raw_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SwarmVerdict:
    """Aggregated verdict for one role swarm."""
    role: str
    verdicts: list[Verdict]
    aggregated_action: list[float]   # confidence-weighted mean across personas
    intra_dissent: list[float]       # per-intervention stdev within this swarm
    mean_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "aggregated_action": self.aggregated_action,
            "intra_dissent": self.intra_dissent,
            "mean_confidence": self.mean_confidence,
        }


@dataclass
class ResonanceReport:
    """
    Cross-swarm result. The 'final_action' is what to recommend; the
    'resonance' and 'dissonance_flags' are the more interesting outputs —
    they tell you which dimensions of the recommendation different
    stakeholder lenses actually agreed on.
    """
    swarm_verdicts: list[SwarmVerdict]
    final_action: list[float]                # length 8
    resonance_per_intervention: list[float]  # length 8, in [0, 1]; higher = more agreement
    dissonance_flags: list[str]              # action names where roles disagreed
    scenario: str
    state_snapshot: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "swarm_verdicts": [s.to_dict() for s in self.swarm_verdicts],
            "final_action": self.final_action,
            "resonance_per_intervention": self.resonance_per_intervention,
            "dissonance_flags": self.dissonance_flags,
            "scenario": self.scenario,
            "state_snapshot": self.state_snapshot,
            "timestamp": self.timestamp,
            "action_names": ACTION_NAMES,
        }
