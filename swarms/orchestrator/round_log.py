from __future__ import annotations
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from swarms.core.verdict import SwarmVerdict, ResonanceReport


@dataclass
class AgentAction:
    """Per-persona action record — analog of MiroFish's AgentAction."""
    round_num: int
    timestamp: str
    role: str
    persona_id: str
    persona_name: str
    action_type: str               # "verdict" / "abstain"
    action_vector: list[float]
    confidence: float
    reasoning: str
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundSummary:
    """One full deliberation round across all swarms."""
    round_num: int
    run_id: str
    start_time: str
    end_time: str | None = None
    scenario: str = ""
    swarms_count: int = 0
    actions_count: int = 0
    actions: list[AgentAction] = field(default_factory=list)
    final_action: list[float] = field(default_factory=list)
    resonance: list[float] = field(default_factory=list)
    dissonance_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_num": self.round_num,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "scenario": self.scenario,
            "swarms_count": self.swarms_count,
            "actions_count": self.actions_count,
            "final_action": self.final_action,
            "resonance": self.resonance,
            "dissonance_flags": self.dissonance_flags,
            "actions": [a.to_dict() for a in self.actions],
        }


class RoundLogger:
    """
    Append-only JSONL writer. One file per `run_id`; each line is a
    completed RoundSummary. Cheap, replayable, and gives the demo a
    real audit trail.
    """

    def __init__(self, log_dir: str | Path | None = None, run_id: str | None = None):
        if log_dir is None:
            log_dir = Path(__file__).resolve().parents[1] / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.log_path = self.log_dir / f"run_{self.run_id}.jsonl"
        self._round_num = 0

    def log_report(self, report: ResonanceReport) -> RoundSummary:
        self._round_num += 1
        start = datetime.now(timezone.utc).isoformat()
        actions = [
            AgentAction(
                round_num=self._round_num,
                timestamp=report.timestamp,
                role=sv.role,
                persona_id=v.persona_id,
                persona_name=v.persona_name,
                action_type="abstain" if v.error else "verdict",
                action_vector=v.action_vector,
                confidence=v.confidence,
                reasoning=v.reasoning,
                success=v.error is None,
                error=v.error,
            )
            for sv in report.swarm_verdicts
            for v in sv.verdicts
        ]
        summary = RoundSummary(
            round_num=self._round_num,
            run_id=self.run_id,
            start_time=start,
            end_time=datetime.now(timezone.utc).isoformat(),
            scenario=report.scenario,
            swarms_count=len(report.swarm_verdicts),
            actions_count=len(actions),
            actions=actions,
            final_action=report.final_action,
            resonance=report.resonance_per_intervention,
            dissonance_flags=report.dissonance_flags,
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
        return summary
