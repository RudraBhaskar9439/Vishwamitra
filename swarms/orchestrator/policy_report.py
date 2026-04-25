"""
LLM-powered IEEE-style policy report generator.

Takes a ResonanceReport (the structured swarm deliberation output) and asks
the LLM to produce a peer-review-quality short paper as a single JSON object.
The strict prompt enforces:

  - Academic third-person voice
  - Specific persona citations and numerical grounding
  - A forward-projection section (what happens if the recommendations ship)
  - A real discussion of methodological limits
  - A banned-phrase list to remove the most common LLM tells

Output JSON keys:
  title, authors, affiliation, abstract, keywords,
  introduction, methodology, results, future_projections, discussion, conclusion
"""
from __future__ import annotations
from typing import Any

from swarms.core.llm_client import LLMClient
from swarms.core.verdict import ACTION_NAMES


SYSTEM_PROMPT = """You are an academic policy analyst writing a short peer-review-quality
paper for an IEEE-style policy analytics venue. The data you are given comes from
a multi-agent LLM deliberation system that simulated four stakeholder swarms
(Student, Teacher, Administrator, Policymaker), each composed of three
heterogeneous personas, and aggregated their verdicts under a confidence-weighted
mean. A cross-swarm Resonance metric (defined as one minus the normalised
standard deviation across the four role-swarm aggregated action vectors) is used
to surface dimensions of disagreement.

Your task is to produce a complete short paper as a SINGLE valid JSON object —
no prose around it, no code fences, no preamble. The JSON must have exactly
these top-level keys, each a multi-paragraph plain-text string:

  title              Declarative title, no marketing phrasing, no colon unless
                     it adds genuine information. Maximum 14 words.
  authors            Use exactly: "Vishwamitra Swarm Deliberation System"
  affiliation        Use exactly: "Multi-Agent Policy Analytics Lab,
                     Educational Commons Research Unit"
  abstract           140-180 words, single paragraph. Problem -> method ->
                     principal finding -> implication. Concrete numbers welcome.
  keywords           5 to 8 comma-separated index terms, lowercase except
                     proper nouns.
  introduction       Three to four paragraphs. Frame the policy problem
                     described in the scenario, the stakes, prior tensions
                     that the deliberation surfaces, and what this paper
                     sets out to determine. Do not summarise the swarm method
                     here -- save that for the methodology section.
  methodology        Two to three paragraphs. Describe the swarm-of-swarms
                     architecture in academic prose: four role-specific
                     swarms, three heterogeneous personas per swarm,
                     persona-conditioned reasoning, confidence-weighted
                     within-swarm aggregation, and the across-swarm Resonance
                     metric. Treat it as a methodology section in a real
                     paper -- describe what was done, not why it is exciting.
  results            Four to five paragraphs. Report the findings using
                     specific numerical evidence: which interventions
                     emerged at high recommended intensity, which Resonance
                     scores indicate consensus versus dissent, and where
                     personas materially diverged. When explaining variance,
                     cite specific personas BY NAME in the form
                     "Maya (first-generation aspirant)" or "Rep. Kumar
                     (union representative)" along with their numerical
                     stance, e.g., "weighted scholarships at 0.78 against
                     Priya's 0.32".
  future_projections Four to five paragraphs forecasting forward. For each
                     intervention recommended at intensity above 0.6,
                     project the likely 6-, 12-, and 24-month system
                     trajectory. Address what plausibly happens if the
                     dissonance-flagged interventions are deployed without
                     resolving the underlying disagreement among
                     stakeholders. Be concrete about second-order effects
                     (teacher attrition cascades, peer-effect dropouts,
                     budget recovery curves). Avoid hedging language.
  discussion         Two to three paragraphs on methodological limits.
                     The personas are LLM-instantiated and may share
                     training-data biases. Confidence values are
                     self-reported. The system cannot validate its own
                     forecasts against ground truth. Identify whose voices
                     this swarm composition likely under-represents.
  conclusion         Two paragraphs. Summarise the recommendation, the
                     conditions under which it holds, and the named
                     decisions that still require human deliberation.

STYLE REQUIREMENTS (strict):

- Third-person academic voice. Past tense for what the deliberation produced;
  present tense for what the data shows; future tense only in the projections
  section.
- Cite specific persona first names with their bracketed role tag whenever
  their reasoning matters to the argument.
- Cite specific numerical values from the deliberation report. Avoid generic
  phrasing like "high agreement" without the number.
- Sentence length must vary. Paragraphs are three to six sentences.
- No bullet lists. No internal subheadings. No "This section ...".
- DO NOT use any of these phrases: "delve into", "delves into", "navigate
  the complexities", "tapestry", "in this ever-evolving", "let us", "dear
  reader", "stand at the precipice", "in conclusion", "at the end of the
  day", "it is important to note that", "it goes without saying", "in
  today's world", "ever-changing landscape", "robust framework",
  "groundbreaking", "revolutionary", "cutting-edge", "leverage" (as a verb),
  "harness the power of", "unlock", "deep dive", "synergy", "paradigm shift".
- Do not editorialise the methodology as transformative or revolutionary.
  Treat it as a tool with documented limits.
- Do not begin sections with "In this section". Do not end with "In summary".
- Do not address the reader. Do not use rhetorical questions.

Output a single JSON object, nothing else.
"""


def _format_data_brief(report: dict[str, Any], state: dict[str, Any], scenario: str) -> str:
    """Compact data dossier handed to the LLM."""
    lines: list[str] = []
    lines.append(f"SCENARIO BRIEF (verbatim from operator):\n{scenario}\n")

    lines.append("OBSERVED SYSTEM STATE AT TIME OF DELIBERATION:")
    for k, v in (state or {}).items():
        if isinstance(v, float):
            lines.append(f"  - {k}: {v:.4f}")
        else:
            lines.append(f"  - {k}: {v}")

    final = report.get("final_action") or [0.0] * 8
    reson = report.get("resonance_per_intervention") or [0.0] * 8
    flags = set(report.get("dissonance_flags") or [])
    lines.append("\nFINAL RECOMMENDED ACTION VECTOR (intensities, range 0.0 to 1.0):")
    for i, n in enumerate(ACTION_NAMES):
        marker = "  [DISSONANT]" if n in flags else ""
        lines.append(f"  - {n}: {final[i]:.3f}    resonance: {reson[i]:.3f}{marker}")

    lines.append(
        "\nDISSONANCE FLAGS (interventions where role-swarm aggregated "
        "recommendations diverged): "
        f"{', '.join(sorted(flags)) if flags else 'none'}"
    )

    lines.append("\nPER-PERSONA VERDICTS:")
    for sv in report.get("swarm_verdicts", []):
        role = sv.get("role", "?")
        mc = sv.get("mean_confidence", 0.0) or 0.0
        lines.append(f"\n  {role.upper()} swarm — mean confidence {mc:.2f}:")
        agg = sv.get("aggregated_action") or [0.0] * 8
        agg_str = ", ".join(f"{ACTION_NAMES[i]}={agg[i]:.2f}" for i in range(len(ACTION_NAMES)))
        lines.append(f"    aggregated_action: [{agg_str}]")
        for v in sv.get("verdicts", []) or []:
            name = v.get("persona_name", "")
            conf = v.get("confidence", 0.0) or 0.0
            reason = (v.get("reasoning") or "").strip().replace("\n", " ")
            vec = v.get("action_vector") or [0.0] * 8
            top3 = sorted(
                [(ACTION_NAMES[i], vec[i]) for i in range(len(ACTION_NAMES))],
                key=lambda x: -x[1],
            )[:3]
            top3_str = ", ".join(f"{n}={val:.2f}" for n, val in top3)
            lines.append(f"    - {name} (conf {conf:.2f}): {reason}")
            lines.append(f"        top recommendations: {top3_str}")
    return "\n".join(lines)


_REQUIRED_KEYS = (
    "title", "authors", "affiliation", "abstract", "keywords",
    "introduction", "methodology", "results", "future_projections",
    "discussion", "conclusion",
)


def _ensure_keys(payload: dict[str, Any]) -> dict[str, str]:
    """Guarantee every required key exists as a string."""
    out: dict[str, str] = {}
    for k in _REQUIRED_KEYS:
        v = payload.get(k, "")
        out[k] = v if isinstance(v, str) else str(v)
    # Nudge fixed-author/affiliation if model deviated.
    if not out["authors"].strip():
        out["authors"] = "Vishwamitra Swarm Deliberation System"
    if not out["affiliation"].strip():
        out["affiliation"] = (
            "Multi-Agent Policy Analytics Lab, Educational Commons Research Unit"
        )
    return out


async def generate_policy_report(
    *,
    report: dict[str, Any],
    state: dict[str, Any],
    scenario: str,
    client: LLMClient | None = None,
) -> dict[str, str]:
    """Run a single LLM call to produce an IEEE-style structured paper."""
    client = client or LLMClient()

    user_prompt = (
        "DELIBERATION DATA TO ANALYSE\n"
        "============================\n\n"
        + _format_data_brief(report, state, scenario)
        + "\n\nProduce the IEEE-style short paper now as a single JSON object "
        "with the keys specified in the system instructions. Do not output "
        "anything else."
    )

    payload = await client.chat_json(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.55,
        max_tokens=4500,
        use_cache=True,
    )
    return _ensure_keys(payload)
