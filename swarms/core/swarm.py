from __future__ import annotations
import asyncio
from typing import Any
from statistics import mean, pstdev

from swarms.core.persona import Persona
from swarms.core.verdict import Verdict, SwarmVerdict, ACTION_NAMES
from swarms.core.swarm_agent import SwarmAgent
from swarms.core.llm_client import LLMClient


class Swarm:
    """
    Role-agnostic swarm. A swarm = a role label + N personas + one shared
    LLM client. Heterogeneity within the swarm comes from each persona's
    system_prompt, NOT from subclassing.

    Adding a new role means adding a new entry to roles.yaml — no Python.
    """

    def __init__(self, role: str, personas: list[Persona], client: LLMClient):
        if not personas:
            raise ValueError(f"Swarm '{role}' must have at least one persona")
        for p in personas:
            if p.role != role:
                raise ValueError(
                    f"Persona {p.id!r} has role={p.role!r} but was added to swarm {role!r}"
                )
        self.role = role
        self.client = client
        self.agents: list[SwarmAgent] = [SwarmAgent(p, client) for p in personas]

    async def deliberate(
        self,
        state_snapshot: dict[str, Any],
        scenario: str,
        action_space_doc: str,
        verdict_instructions: str,
        *,
        model: str | None = None,
        persona_weights: dict[str, float] | None = None,
        persona_fits: list[dict[str, Any]] | None = None,
    ) -> SwarmVerdict:
        """All N personas deliberate in parallel.

        Parameters
        ----------
        model : str | None
            L1 orchestrator's model assignment for this swarm — forwarded
            to every agent in the swarm.
        persona_weights : dict[str, float] | None
            L2 orchestrator's per-persona weight (persona_id → weight).
            Used inside _aggregate to multiply confidence × fit_weight.
            None means equal weight per persona (legacy behaviour).
        persona_fits : list[dict] | None
            Full PersonaFitDecision dicts to attach to the SwarmVerdict
            for downstream UI/audit. Optional.
        """
        verdicts: list[Verdict] = await asyncio.gather(
            *[
                a.deliberate(
                    state_snapshot=state_snapshot,
                    scenario=scenario,
                    action_space_doc=action_space_doc,
                    verdict_instructions=verdict_instructions,
                    model=model,
                )
                for a in self.agents
            ],
            return_exceptions=False,
        )
        return self._aggregate(verdicts, persona_weights or {}, persona_fits or [])

    # ------------------------- aggregation -------------------------
    def _aggregate(
        self,
        verdicts: list[Verdict],
        persona_weights: dict[str, float],
        persona_fits: list[dict[str, Any]],
    ) -> SwarmVerdict:
        # Drop verdicts that errored out (zero-confidence abstentions).
        live = [v for v in verdicts if v.error is None and v.confidence > 0.0]
        n_actions = len(ACTION_NAMES)

        if not live:
            return SwarmVerdict(
                role=self.role,
                verdicts=verdicts,
                aggregated_action=[0.0] * n_actions,
                intra_dissent=[0.0] * n_actions,
                mean_confidence=0.0,
                persona_fits=persona_fits,
            )

        # Combined weight = self-reported confidence × L2 fit weight.
        # Fit weight defaults to 1.0 when no L2 decision was supplied.
        combined = [
            v.confidence * float(persona_weights.get(v.persona_id, 1.0))
            for v in live
        ]
        wsum = sum(combined) or 1.0
        agg = [
            sum(v.action_vector[i] * w for v, w in zip(live, combined)) / wsum
            for i in range(n_actions)
        ]

        # Within-swarm dissent = stdev across personas, per dimension —
        # UNWEIGHTED so it surfaces raw disagreement among the lenses,
        # independent of who the L2 router thinks should be loud.
        if len(live) >= 2:
            dissent = [
                pstdev([v.action_vector[i] for v in live]) for i in range(n_actions)
            ]
        else:
            dissent = [0.0] * n_actions

        # mean_confidence stays unweighted — it's the swarm's "voice strength"
        # for cross-swarm aggregation, conceptually distinct from fit.
        return SwarmVerdict(
            role=self.role,
            verdicts=verdicts,
            aggregated_action=agg,
            intra_dissent=dissent,
            mean_confidence=mean([v.confidence for v in live]),
            persona_fits=persona_fits,
        )
