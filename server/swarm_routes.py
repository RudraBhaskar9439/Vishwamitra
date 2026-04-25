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


class DeliberateResponse(BaseModel):
    final_action: list[float]
    action_names: list[str]
    resonance_per_intervention: list[float]
    dissonance_flags: list[str]
    swarm_verdicts: list[dict[str, Any]]
    scenario: str
    timestamp: str


# ------------------------------ routes ----------------------------
@router.get("/info")
def info() -> dict[str, Any]:
    """Show which roles, personas, and model are configured."""
    try:
        return get_manager().describe()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"manager init failed: {e}")


@router.post("/deliberate", response_model=DeliberateResponse)
async def deliberate(req: DeliberateRequest) -> DeliberateResponse:
    """Run all swarms on the given state snapshot and return a ResonanceReport."""
    try:
        manager = get_manager()
        report = await manager.deliberate(req.state, scenario=req.scenario)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"deliberation failed: {e}")

    payload = report.to_dict()
    return DeliberateResponse(
        final_action=payload["final_action"],
        action_names=payload["action_names"],
        resonance_per_intervention=payload["resonance_per_intervention"],
        dissonance_flags=payload["dissonance_flags"],
        swarm_verdicts=payload["swarm_verdicts"],
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


class PolicyReportResponse(BaseModel):
    title: str = ""
    authors: str = "Vishwamitra Swarm Deliberation System"
    affiliation: str = "Multi-Agent Policy Analytics Lab, Educational Commons Research Unit"
    abstract: str = ""
    keywords: str = ""
    introduction: str = ""
    methodology: str = ""
    results: str = ""
    future_projections: str = ""
    discussion: str = ""
    conclusion: str = ""


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
