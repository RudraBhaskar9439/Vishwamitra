"""
LLM-powered Educational Policymaking Brief generator — IN-DEPTH version.

Produces a comprehensive policy brief in TWO sequential LLM calls (so each
call fits comfortably under per-minute token windows), then merges the
output into a single structured response. The schema is significantly
deeper than v1: an executive summary, an operational scenario analysis,
a four-phase implementation roadmap, a risk register, and a stakeholder-
feedback synthesis with per-persona direct quotes.

Output JSON schema (after merge):

  Strategic Context  (Call 1)
  -----------------------------------
  title, subtitle, executive_summary, what_is
  operational_context, state_diagnostic[*], root_cause_hypothesis,
  stakes_of_inaction, success_criteria[]

  Six-Stage Process  (Call 1 stages 1-3, Call 2 stages 4-6)
  -----------------------------------
  stage_1_description, stage_1_bullets
  stage_2_description, stage_2_bullets, stage_2_influencers
  stage_3_description, stage_3_bullets, stage_3_contributors
  stage_4_description, stage_4_bullets
  stage_5_description, stage_5_bullets, stage_5_challenges
  stage_6_description, stage_6_bullets

  Implementation & Risk  (Call 2)
  -----------------------------------
  roadmap[*]              4 phases × {name, window, objective, actions[],
                                       owners[], milestones[], dependencies[]}
  risk_register[*]        items × {risk, likelihood, impact, mitigation}

  Stakeholders & Feedback  (Call 2)
  -----------------------------------
  stakeholders[*], persona_feedback[*]
  areas_of_agreement[], areas_of_contention[]
  iterative_nature[], challenges[], strategies[], takeaway

The report uses an explicitly heavier model (Llama 3.3 70B by default)
because depth and prose quality matter here more than throughput.
"""
from __future__ import annotations
import os
from typing import Any

from swarms.core.llm_client import LLMClient
from swarms.core.verdict import ACTION_NAMES


# Lightweight model — fast and high quota. Override via REPORT_MODEL_NAME.
DEFAULT_REPORT_MODEL = "llama-3.1-8b-instant"


# ============================================================================
# SHARED STYLE GUARDRAILS (compact — fits inside small-model TPM budgets)
# ============================================================================
_STYLE_GUARDRAILS = """
STYLE:
  - Third-person, declarative, evidence-grounded.
  - Cite personas by name + tag: "Maya (first-gen)", "Sharma (22-yr veteran)",
    "Verma (fiscal hawk)". Quote specific numbers from the deliberation.
  - Bullets are full sentences (1-2 sentences each), not fragments.
  - BANNED: delve, navigate the complexities, tapestry, ever-evolving,
    in conclusion, at the end of the day, leverage (verb), groundbreaking,
    transformative, paradigm shift, deep dive, intricacies, harness the
    power of, robust framework, synergy.
  - Don't say "the swarm" alone — use "the Teacher swarm" or "the deliberation".
  - Don't address the reader. No rhetorical questions.

OUTPUT: single FLAT JSON object. No prose, no code fences.
"""


# ============================================================================
# CALL 1 — Strategic Context + Stages 1-3
# ============================================================================
SYSTEM_PROMPT_PART_1 = """Produce part 1 of an Educational Policy Brief from
a Vishwamitra swarm deliberation. Output a SINGLE FLAT JSON object — every
listed key is top-level. No nested wrappers. No code fences.

Required FLAT top-level keys:

  title              "Educational Policy Brief: <Specific Crisis>" (≤14 words)
  subtitle           1 evocative line.
  executive_summary  4-5 sentences: crisis, deliberation, recommendation,
                     decisions still belonging to humans.
  what_is            4-5 sentences on educational policy in THIS crisis.

  operational_context     2 paragraphs, read like an incident report,
                          citing state numbers verbatim.
  state_diagnostic        list of 5-7 objects, each:
                          {"metric": <field name>,
                           "value":  <formatted value>,
                           "interpretation": <1 sentence>}
                          Order by severity.
  root_cause_hypothesis   2 paragraphs on why the system arrived here.
                          Link upstream causes to downstream symptoms.
  stakes_of_inaction      2 paragraphs on 90/180-day cascade if no action.
                          Reference cascade dynamics (teacher exits →
                          class size → burnout).
  success_criteria        4-6 measurable bullets, each naming a metric +
                          target movement.

  stage_1_description     2-3 sentences citing 2 state numbers.
  stage_1_bullets         4-5 bullets with numbers.
  stage_2_description     2-3 sentences on agenda dynamics.
  stage_2_bullets         4-5 bullets citing personas by name.
  stage_2_influencers     1 sentence.
  stage_3_description     2-3 sentences on intervention formulation.
  stage_3_bullets         5-6 bullets — EACH one intervention with its
                          intensity number + persona-grounded rationale.
  stage_3_contributors    1 sentence naming swarms + 1-2 key personas.

""" + _STYLE_GUARDRAILS


# ============================================================================
# CALL 2 — Stages 4-6 + Roadmap + Risk + Stakeholders + Feedback Synthesis
# ============================================================================
SYSTEM_PROMPT_PART_2 = """Produce part 2 of an Educational Policy Brief from
a Vishwamitra swarm deliberation. Part 1 (executive, scenario analysis,
stages 1-3) was authored elsewhere — stay consistent with its scenario,
personas, and numbers. Output a SINGLE FLAT JSON object — every key is
top-level. No nested wrappers. No code fences.

Required FLAT top-level keys:

  stage_4_description     2-3 sentences on adoption tensions.
  stage_4_bullets         4-5 bullets pairing recommendations with persona
                          tensions.
  stage_5_description     2-3 sentences on implementation challenges.
  stage_5_bullets         4-5 bullets on capacity / timeline / sequencing.
  stage_5_challenges      1 sentence on primary implementation risks.
  stage_6_description     2-3 sentences on outcome measurement.
  stage_6_bullets         3-4 bullets — each = state-vector metric + target
                          direction + timeline.

  roadmap                 EXACTLY 4 phase objects:
                          {"phase_name": "Phase N: <Verb-led name>",
                           "window":     "0-30 days" | "30-90 days" | etc.,
                           "objective":  <1 sentence>,
                           "actions":    [3-4 concrete actions],
                           "owners":     [2-3 named owner roles],
                           "milestones": [2-3 quantified milestones],
                           "dependencies": [1-2 dependencies]}
                          Phases: (1) stabilise 0-30d, (2) rollout 30-90d,
                          (3) course-correct 3-6mo, (4) evaluate 6-12mo.

  risk_register           4-5 risk objects:
                          {"risk": str, "likelihood": "low"|"medium"|"high",
                           "impact": "low"|"medium"|"high", "mitigation": str}
                          Cover political, fiscal, capacity, equity,
                          behavioural cascade. At least one per dissonance
                          flag.

""" + _STYLE_GUARDRAILS


# ============================================================================
# CALL 3 — Stakeholders + feedback synthesis + closing
# ============================================================================
SYSTEM_PROMPT_PART_3 = """Produce part 3 of an Educational Policy Brief from
a Vishwamitra swarm deliberation. Parts 1 and 2 (strategic context, stages,
roadmap, risk register) were authored elsewhere — stay consistent with
their scenario, personas, and numbers. Output a SINGLE FLAT JSON object.
No nested wrappers. No code fences.

Required FLAT top-level keys:

  stakeholders            EXACTLY 6 objects {"name": str, "role": str}.
                          First 4: Student Body, Teaching Staff, School
                          Administration, Policymakers. The role field
                          summarises that swarm's contribution in THIS
                          deliberation, naming at least one persona. Add
                          2 scenario-relevant extras (Parents/Community,
                          District Auditor, etc.).

  persona_feedback        6-8 objects, ≥1 per role:
                          {"persona_name": str, "role": str,
                           "key_concern":  <1 sentence>,
                           "direct_quote": <2-3 sentences in their voice>,
                           "actionable_request": <1 sentence>}

  areas_of_agreement      3-5 bullets where lenses converged. Cite
                          resonance scores when relevant.
  areas_of_contention     3-5 bullets where lenses diverged. One per
                          dissonance flag, naming flagged intervention
                          and the underlying persona tension.

  iterative_nature        3-4 bullets on what feeds back into next cycle.
  challenges              4-5 scenario-specific challenges.
  strategies              4-5 actionable bullets.
  takeaway                2 paragraphs: synthesis, then which calls
                          require human judgement.

""" + _STYLE_GUARDRAILS


# ============================================================================
# DATA BRIEF (shared input)
# ============================================================================
_REASON_CHARS = 220
_KEY_STATE_FIELDS = (
    "enrollment_rate", "attendance_rate", "dropout_rate",
    "teacher_retention", "teacher_burnout", "student_engagement",
    "resource_allocation", "avg_class_size", "budget_remaining", "trust_score",
)


def _format_data_brief(report: dict[str, Any], state: dict[str, Any], scenario: str) -> str:
    lines: list[str] = []
    lines.append(f"SCENARIO: {scenario}\n")

    lines.append("STATE:")
    state = state or {}
    for k in _KEY_STATE_FIELDS:
        if k in state:
            v = state[k]
            lines.append(f"  {k}={v:.3f}" if isinstance(v, float) else f"  {k}={v}")

    final = report.get("final_action") or [0.0] * 8
    reson = report.get("resonance_per_intervention") or [0.0] * 8
    flags = set(report.get("dissonance_flags") or [])

    lines.append("\nRECOMMENDED ACTION (intensity / resonance):")
    for i, n in enumerate(ACTION_NAMES):
        flag = "  DISSONANT" if n in flags else ""
        lines.append(f"  {n}: {final[i]:.2f} / {reson[i]:.2f}{flag}")

    if flags:
        lines.append(f"\nDISSONANCE_FLAGS: {', '.join(sorted(flags))}")

    lines.append("\nPERSONA VERDICTS:")
    for sv in report.get("swarm_verdicts", []):
        role = sv.get("role", "?")
        mc = sv.get("mean_confidence", 0.0) or 0.0
        lines.append(f"\n[{role.upper()}] mean_conf={mc:.2f}")
        for v in sv.get("verdicts", []) or []:
            name = v.get("persona_name", "")
            conf = v.get("confidence", 0.0) or 0.0
            reason = (v.get("reasoning") or "").strip().replace("\n", " ")
            if len(reason) > _REASON_CHARS:
                reason = reason[:_REASON_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(f"  - {name} (c={conf:.2f}): {reason}")
    return "\n".join(lines)


# ============================================================================
# Output normalisation
# ============================================================================
_STR_KEYS = (
    # strategic context
    "title", "subtitle", "executive_summary", "what_is",
    "operational_context", "root_cause_hypothesis", "stakes_of_inaction",
    # stages
    "stage_1_description", "stage_2_description", "stage_2_influencers",
    "stage_3_description", "stage_3_contributors",
    "stage_4_description",
    "stage_5_description", "stage_5_challenges",
    "stage_6_description",
    # final
    "takeaway",
)
_LIST_OF_STR_KEYS = (
    "success_criteria",
    "stage_1_bullets", "stage_2_bullets", "stage_3_bullets",
    "stage_4_bullets", "stage_5_bullets", "stage_6_bullets",
    "iterative_nature",
    "areas_of_agreement", "areas_of_contention",
    "challenges", "strategies",
)


def _coerce_str(v: Any) -> str:
    return v if isinstance(v, str) else str(v) if v is not None else ""


def _coerce_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        parts = [p.strip(" -•\t").strip() for p in v.splitlines() if p.strip()]
        return [p for p in parts if p]
    return []


def _coerce_state_diagnostic(v: Any) -> list[dict[str, str]]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if isinstance(item, dict):
            out.append({
                "metric": _coerce_str(item.get("metric")),
                "value": _coerce_str(item.get("value")),
                "interpretation": _coerce_str(item.get("interpretation")),
            })
    return out


def _coerce_roadmap(v: Any) -> list[dict[str, Any]]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if isinstance(item, dict):
            out.append({
                "phase_name": _coerce_str(item.get("phase_name")),
                "window": _coerce_str(item.get("window")),
                "objective": _coerce_str(item.get("objective")),
                "actions": _coerce_str_list(item.get("actions")),
                "owners": _coerce_str_list(item.get("owners")),
                "milestones": _coerce_str_list(item.get("milestones")),
                "dependencies": _coerce_str_list(item.get("dependencies")),
            })
    return out


def _coerce_risk_register(v: Any) -> list[dict[str, str]]:
    if not isinstance(v, list):
        return []
    out = []
    valid_levels = {"low", "medium", "high"}
    for item in v:
        if isinstance(item, dict):
            lk = _coerce_str(item.get("likelihood")).lower()
            ip = _coerce_str(item.get("impact")).lower()
            out.append({
                "risk": _coerce_str(item.get("risk")),
                "likelihood": lk if lk in valid_levels else "medium",
                "impact": ip if ip in valid_levels else "medium",
                "mitigation": _coerce_str(item.get("mitigation")),
            })
    return out


def _coerce_stakeholders(v: Any) -> list[dict[str, str]]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if isinstance(item, dict):
            name = _coerce_str(item.get("name") or item.get("stakeholder"))
            role = _coerce_str(item.get("role") or item.get("contribution"))
            if name or role:
                out.append({"name": name, "role": role})
    return out


def _coerce_persona_feedback(v: Any) -> list[dict[str, str]]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if isinstance(item, dict):
            out.append({
                "persona_name": _coerce_str(item.get("persona_name")),
                "role": _coerce_str(item.get("role")),
                "key_concern": _coerce_str(item.get("key_concern")),
                "direct_quote": _coerce_str(item.get("direct_quote")),
                "actionable_request": _coerce_str(item.get("actionable_request")),
            })
    return out


_KNOWN_KEYS = (
    set(_STR_KEYS)
    | set(_LIST_OF_STR_KEYS)
    | {"state_diagnostic", "roadmap", "risk_register", "stakeholders", "persona_feedback"}
)


def _flatten_response(payload: Any, depth: int = 0) -> dict[str, Any]:
    """Recursively flatten a dict so that any nested dict whose own keys do
    NOT match our schema (e.g., the model wrapped fields under category
    labels like "EXECUTIVE LAYER") gets merged up into the parent.

    A nested dict whose keys ARE our known schema fields is treated as
    a real container and its entries lifted to top level.
    """
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k in _KNOWN_KEYS:
            # Real schema field — keep as-is.
            out[k] = v
            continue
        if isinstance(v, dict):
            # Unrecognised parent — flatten its children.
            inner = _flatten_response(v, depth + 1)
            for ik, iv in inner.items():
                if ik not in out:
                    out[ik] = iv
        # Else: silently drop unknown scalar/list keys to avoid junk.
    return out


def _ensure_keys(merged: dict[str, Any]) -> dict[str, Any]:
    # First pass: lift fields out of any category-label wrappers the LLM
    # might have produced.
    flat = _flatten_response(merged)

    out: dict[str, Any] = {}
    for k in _STR_KEYS:
        out[k] = _coerce_str(flat.get(k, ""))
    for k in _LIST_OF_STR_KEYS:
        out[k] = _coerce_str_list(flat.get(k))
    out["state_diagnostic"] = _coerce_state_diagnostic(flat.get("state_diagnostic"))
    out["roadmap"] = _coerce_roadmap(flat.get("roadmap"))
    out["risk_register"] = _coerce_risk_register(flat.get("risk_register"))
    out["stakeholders"] = _coerce_stakeholders(flat.get("stakeholders"))
    out["persona_feedback"] = _coerce_persona_feedback(flat.get("persona_feedback"))
    return out


# ============================================================================
# Main entry point — runs both calls and merges
# ============================================================================
async def generate_policy_report(
    *,
    report: dict[str, Any],
    state: dict[str, Any],
    scenario: str,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Two LLM calls → merged Educational Policy Brief.

    Both calls explicitly request the heavyweight model
    (default: llama-3.3-70b-versatile) because narrative quality matters
    here more than throughput. Override with REPORT_MODEL_NAME env var.
    """
    client = client or LLMClient()
    report_model = os.getenv("REPORT_MODEL_NAME") or DEFAULT_REPORT_MODEL

    data_brief = _format_data_brief(report, state, scenario)

    def _user(part_label: str, what: str) -> str:
        return (
            "DELIBERATION DATA\n"
            "=================\n\n"
            + data_brief
            + f"\n\nProduce {part_label} ({what}) now as a single FLAT JSON "
              "object with the keys specified. Stay consistent with the "
              "other parts in tone, persona references, and numbers."
        )

    # Tuned for Llama 3.1 8B-instant's 6K TPM. Three calls of ~1800 output
    # tokens each, with ~600 token input, fit safely under per-request
    # and per-minute caps and let every requested key be authored fully.
    part_1 = await client.chat_json(
        system=SYSTEM_PROMPT_PART_1,
        user=_user("PART 1", "strategic context + stages 1-3"),
        temperature=0.5,
        max_tokens=1900,
        use_cache=True,
        model=report_model,
    )
    part_2 = await client.chat_json(
        system=SYSTEM_PROMPT_PART_2,
        user=_user("PART 2", "stages 4-6, implementation roadmap, risk register"),
        temperature=0.5,
        max_tokens=1900,
        use_cache=True,
        model=report_model,
    )
    part_3 = await client.chat_json(
        system=SYSTEM_PROMPT_PART_3,
        user=_user("PART 3", "stakeholders, persona feedback, consensus, closing"),
        temperature=0.5,
        max_tokens=1900,
        use_cache=True,
        model=report_model,
    )
    merged = {**part_1, **part_2, **part_3}
    return _ensure_keys(merged)
