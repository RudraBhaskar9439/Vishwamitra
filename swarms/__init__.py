"""
VIDYA Swarms — multi-perspective LLM deliberation layer.

Four role swarms (student, teacher, admin, policymaker) each composed of
N heterogeneous personas. Verdicts are aggregated within each swarm, then
across swarms into a ResonanceReport that surfaces both consensus and
dissonance for any educational policy decision.

Public entry point:
    from swarms import SwarmManager
    manager = SwarmManager()
    report = await manager.deliberate(state_dict, scenario="funding_cut")
"""

from swarms.orchestrator.swarm_manager import SwarmManager
from swarms.orchestrator.router import (
    WeightAllocator, RouterDecision, OrchestratorPlan,
    PersonaAllocator, PersonaFitDecision,
)
from swarms.core.verdict import Verdict, SwarmVerdict, ResonanceReport
from swarms.core.persona import Persona

__all__ = [
    "SwarmManager",
    "WeightAllocator",
    "RouterDecision",
    "OrchestratorPlan",
    "PersonaAllocator",
    "PersonaFitDecision",
    "Verdict",
    "SwarmVerdict",
    "ResonanceReport",
    "Persona",
]
