from __future__ import annotations
from statistics import pstdev

from swarms.core.verdict import SwarmVerdict, ResonanceReport, ACTION_NAMES


# Per-intervention agreement is computed as 1 - normalized_stdev across
# the role swarms' aggregated action vectors. We normalize by the maximum
# possible stdev for values in [0, 1], which is 0.5 (when half are 0 and
# half are 1). Values clamp to [0, 1].
_MAX_STD_UNIT = 0.5


def compute_resonance(
    swarm_verdicts: list[SwarmVerdict],
    *,
    dissonance_threshold: float = 0.55,
    scenario: str,
    state_snapshot: dict,
    swarm_weights: dict[str, float] | None = None,
) -> ResonanceReport:
    """
    Aggregate role-swarm verdicts into a final action vector + resonance map.

    For each of the 8 interventions:
      - final value = confidence-weighted mean of swarm-aggregated values,
                      multiplied by the orchestrator's per-role verdict
                      weight if `swarm_weights` is provided
      - resonance   = 1 - (stdev across swarms / 0.5), clipped to [0, 1]
                      — DELIBERATELY UNWEIGHTED so it still reports raw
                      disagreement among lenses regardless of how much
                      we trust each lens
      - flagged as dissonant if resonance < dissonance_threshold

    `swarm_weights` is a mapping role_name -> verdict_weight. Roles not in
    the mapping default to weight 1.0.
    """
    n = len(ACTION_NAMES)
    swarm_weights = swarm_weights or {}

    if not swarm_verdicts:
        return ResonanceReport(
            swarm_verdicts=[],
            final_action=[0.0] * n,
            resonance_per_intervention=[0.0] * n,
            dissonance_flags=[],
            scenario=scenario,
            state_snapshot=state_snapshot,
        )

    # Confidence-weighted mean across swarms, multiplied by orchestrator's
    # role verdict weight. Higher orchestrator weight pulls the final
    # action toward that role's recommendation.
    weights = [
        max(sv.mean_confidence, 1e-6) * float(swarm_weights.get(sv.role, 1.0))
        for sv in swarm_verdicts
    ]
    wsum = sum(weights) or 1.0
    final = [
        sum(sv.aggregated_action[i] * w for sv, w in zip(swarm_verdicts, weights)) / wsum
        for i in range(n)
    ]

    # Resonance = 1 - normalized stdev across swarms (UNWEIGHTED — pure
    # disagreement signal, independent of how much we trust each lens).
    if len(swarm_verdicts) >= 2:
        resonance: list[float] = []
        for i in range(n):
            vals = [sv.aggregated_action[i] for sv in swarm_verdicts]
            std = pstdev(vals)
            r = 1.0 - min(std / _MAX_STD_UNIT, 1.0)
            resonance.append(max(0.0, min(1.0, r)))
    else:
        resonance = [1.0] * n  # only one swarm -> trivially resonant

    flags = [
        ACTION_NAMES[i]
        for i in range(n)
        if resonance[i] < dissonance_threshold
    ]

    return ResonanceReport(
        swarm_verdicts=swarm_verdicts,
        final_action=final,
        resonance_per_intervention=resonance,
        dissonance_flags=flags,
        scenario=scenario,
        state_snapshot=state_snapshot,
    )
