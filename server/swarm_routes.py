"""
FastAPI routes for the Vishwamitra swarm layer.

Exposes a single deliberation endpoint that runs all 4 role swarms in
parallel and returns a ResonanceReport (final action vector + per-
intervention agreement scores + dissonance flags).
"""
from __future__ import annotations
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from swarms import SwarmManager


router = APIRouter(prefix="/swarms", tags=["swarms"])


# Lazy singleton — first request initializes, subsequent reuse.
_manager: SwarmManager | None = None


def get_manager() -> SwarmManager:
    global _manager
    if _manager is None:
        _manager = SwarmManager()
    return _manager


# ----------------------- request / response -----------------------
class OrchestratorRoleOverride(BaseModel):
    model: Optional[str] = None
    verdict_weight: Optional[float] = None


class OrchestratorOverrides(BaseModel):
    """Per-role manual override for the WeightAllocator.

    Shape:
      { "mode": "auto" | "manual",
        "roles": { "<role>": { "model": "...", "verdict_weight": 1.5 } } }

    Any role not present in `roles` falls back to the auto heuristic.
    """
    mode: Optional[str] = None
    roles: dict[str, OrchestratorRoleOverride] = Field(default_factory=dict)


class DeliberateRequest(BaseModel):
    state: dict[str, Any] = Field(
        ...,
        description="System state snapshot. Free-form key/value; SystemState fields are recommended.",
        examples=[{
            "enrollment_rate": 0.62,
            "attendance_rate": 0.55,
            "dropout_rate": 0.28,
            "teacher_retention": 0.71,
            "teacher_burnout": 0.65,
            "budget_remaining": 420000.0,
            "step": 12,
        }],
    )
    scenario: str = Field("general", description="Short scenario name or description.")
    orchestrator_overrides: Optional[OrchestratorOverrides] = Field(
        default=None,
        description="Optional per-role overrides for the dynamic router. "
                    "Roles not specified fall back to the auto heuristic.",
    )


class DeliberateResponse(BaseModel):
    final_action: list[float]
    action_names: list[str]
    resonance_per_intervention: list[float]
    dissonance_flags: list[str]
    swarm_verdicts: list[dict[str, Any]]
    orchestrator_plan: Optional[dict[str, Any]] = None
    scenario: str
    timestamp: str


class PreviewPlanRequest(BaseModel):
    state: dict[str, Any]
    orchestrator_overrides: Optional[OrchestratorOverrides] = None


# ------------------------------ routes ----------------------------
@router.get("/info")
def info() -> dict[str, Any]:
    """Show which roles, personas, and model are configured."""
    try:
        return get_manager().describe()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"manager init failed: {e}")


def _normalise_overrides(
    body: OrchestratorOverrides | None,
) -> dict[str, Any] | None:
    """Pydantic → plain dict shape WeightAllocator.allocate expects."""
    if body is None:
        return None
    roles_dict: dict[str, dict[str, Any]] = {}
    for role, ov in (body.roles or {}).items():
        entry: dict[str, Any] = {}
        if ov.model is not None:
            entry["model"] = ov.model
        if ov.verdict_weight is not None:
            entry["verdict_weight"] = float(ov.verdict_weight)
        if entry:
            roles_dict[role] = entry
    if not roles_dict and not body.mode:
        return None
    return {"mode": body.mode or "auto", "roles": roles_dict}


@router.post("/preview-plan")
def preview_plan(req: PreviewPlanRequest) -> dict[str, Any]:
    """Preview what the router WOULD allocate for this state — no LLM calls.

    Used by the UI to render the OrchestratorPanel reactively as sliders move.
    """
    try:
        manager = get_manager()
        return manager.preview_plan(
            req.state,
            _normalise_overrides(req.orchestrator_overrides),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"preview failed: {e}")


@router.post("/deliberate", response_model=DeliberateResponse)
async def deliberate(req: DeliberateRequest) -> DeliberateResponse:
    """Run all swarms on the given state snapshot and return a ResonanceReport."""
    try:
        manager = get_manager()
        report = await manager.deliberate(
            req.state,
            scenario=req.scenario,
            orchestrator_overrides=_normalise_overrides(req.orchestrator_overrides),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"deliberation failed: {e}")

    payload = report.to_dict()
    return DeliberateResponse(
        final_action=payload["final_action"],
        action_names=payload["action_names"],
        resonance_per_intervention=payload["resonance_per_intervention"],
        dissonance_flags=payload["dissonance_flags"],
        swarm_verdicts=payload["swarm_verdicts"],
        orchestrator_plan=payload.get("orchestrator_plan"),
        scenario=payload["scenario"],
        timestamp=payload["timestamp"],
    )


# ----------------------- IEEE-style policy report -----------------------
class PolicyReportRequest(BaseModel):
    """Body for /policy-report. Pass the full ResonanceReport JSON, the
    state snapshot the deliberation ran on, and the operator scenario brief."""
    report: dict[str, Any] = Field(..., description="ResonanceReport JSON returned by /deliberate")
    state: dict[str, Any] = Field(default_factory=dict)
    scenario: str = Field("", description="Operator scenario brief, verbatim.")


class Stakeholder(BaseModel):
    name: str = ""
    role: str = ""


class StateDiagnosticItem(BaseModel):
    metric: str = ""
    value: str = ""
    interpretation: str = ""


class RoadmapPhase(BaseModel):
    phase_name: str = ""
    window: str = ""
    objective: str = ""
    actions: list[str] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class RiskItem(BaseModel):
    risk: str = ""
    likelihood: str = "medium"
    impact: str = "medium"
    mitigation: str = ""


class PersonaFeedback(BaseModel):
    persona_name: str = ""
    role: str = ""
    key_concern: str = ""
    direct_quote: str = ""
    actionable_request: str = ""


class PolicyReportResponse(BaseModel):
    # Executive layer
    title: str = ""
    subtitle: str = ""
    executive_summary: str = ""
    what_is: str = ""

    # Scenario analysis (NEW)
    operational_context: str = ""
    state_diagnostic: list[StateDiagnosticItem] = Field(default_factory=list)
    root_cause_hypothesis: str = ""
    stakes_of_inaction: str = ""
    success_criteria: list[str] = Field(default_factory=list)

    # Six-stage process
    stage_1_description: str = ""
    stage_1_bullets: list[str] = Field(default_factory=list)

    stage_2_description: str = ""
    stage_2_bullets: list[str] = Field(default_factory=list)
    stage_2_influencers: str = ""

    stage_3_description: str = ""
    stage_3_bullets: list[str] = Field(default_factory=list)
    stage_3_contributors: str = ""

    stage_4_description: str = ""
    stage_4_bullets: list[str] = Field(default_factory=list)

    stage_5_description: str = ""
    stage_5_bullets: list[str] = Field(default_factory=list)
    stage_5_challenges: str = ""

    stage_6_description: str = ""
    stage_6_bullets: list[str] = Field(default_factory=list)

    # Implementation & risk (NEW)
    roadmap: list[RoadmapPhase] = Field(default_factory=list)
    risk_register: list[RiskItem] = Field(default_factory=list)

    # Stakeholders & feedback synthesis (NEW)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    persona_feedback: list[PersonaFeedback] = Field(default_factory=list)
    areas_of_agreement: list[str] = Field(default_factory=list)
    areas_of_contention: list[str] = Field(default_factory=list)

    # Final
    iterative_nature: list[str] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    takeaway: str = ""


@router.post("/policy-report", response_model=PolicyReportResponse)
async def policy_report(req: PolicyReportRequest) -> PolicyReportResponse:
    """Generate an IEEE-style narrative policy paper from a ResonanceReport.

    This is a single LLM call that produces structured prose: abstract,
    introduction, methodology, results/analysis, future projections,
    discussion, conclusion. It does NOT re-run the deliberation; pass the
    report from /deliberate.
    """
    from swarms.orchestrator.policy_report import generate_policy_report
    try:
        manager = get_manager()
        out = await generate_policy_report(
            report=req.report,
            state=req.state,
            scenario=req.scenario,
            client=manager.client,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"policy report generation failed: {e}",
        )
    return PolicyReportResponse(**out)
