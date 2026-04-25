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
) -> ResonanceReport:
    """
    Aggregate role-swarm verdicts into a final action vector + resonance map.

    For each of the 8 interventions:
      - final value = confidence-weighted mean of swarm-aggregated values
      - resonance   = 1 - (stdev across swarms / 0.5), clipped to [0, 1]
      - flagged as dissonant if resonance < dissonance_threshold
    """
    n = len(ACTION_NAMES)

    if not swarm_verdicts:
        return ResonanceReport(
            swarm_verdicts=[],
            final_action=[0.0] * n,
            resonance_per_intervention=[0.0] * n,
            dissonance_flags=[],
            scenario=scenario,
            state_snapshot=state_snapshot,
        )

    # Confidence-weighted mean across swarms, per intervention.
    weights = [max(sv.mean_confidence, 1e-6) for sv in swarm_verdicts]
    wsum = sum(weights)
    final = [
        sum(sv.aggregated_action[i] * w for sv, w in zip(swarm_verdicts, weights)) / wsum
        for i in range(n)
    ]

    # Resonance = 1 - normalized stdev across swarms.
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
