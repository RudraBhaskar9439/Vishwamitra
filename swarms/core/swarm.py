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
    ) -> SwarmVerdict:
        verdicts: list[Verdict] = await asyncio.gather(
            *[
                a.deliberate(
                    state_snapshot=state_snapshot,
                    scenario=scenario,
                    action_space_doc=action_space_doc,
                    verdict_instructions=verdict_instructions,
                )
                for a in self.agents
            ],
            return_exceptions=False,
        )
        return self._aggregate(verdicts)

    # ------------------------- aggregation -------------------------
    def _aggregate(self, verdicts: list[Verdict]) -> SwarmVerdict:
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
            )

        # Confidence-weighted mean per intervention dimension.
        weights = [v.confidence for v in live]
        wsum = sum(weights) or 1.0
        agg = [
            sum(v.action_vector[i] * v.confidence for v in live) / wsum
            for i in range(n_actions)
        ]

        # Within-swarm dissent = stdev across personas, per dimension.
        if len(live) >= 2:
            dissent = [
                pstdev([v.action_vector[i] for v in live]) for i in range(n_actions)
            ]
        else:
            dissent = [0.0] * n_actions

        return SwarmVerdict(
            role=self.role,
            verdicts=verdicts,
            aggregated_action=agg,
            intra_dissent=dissent,
            mean_confidence=mean(weights),
        )
