from __future__ import annotations
from typing import Any

from swarms.core.persona import Persona
from swarms.core.verdict import Verdict, ACTION_NAMES
from swarms.core.llm_client import LLMClient


class SwarmAgent:
    """
    A single LLM-backed deliberator. One persona, one verdict per call.
    The role-specific behavior comes entirely from `persona.system_prompt`;
    this class is role-agnostic.
    """

    def __init__(self, persona: Persona, client: LLMClient):
        self.persona = persona
        self.client = client

    async def deliberate(
        self,
        state_snapshot: dict[str, Any],
        scenario: str,
        action_space_doc: str,
        verdict_instructions: str,
    ) -> Verdict:
        user_prompt = self._build_user_prompt(
            state_snapshot=state_snapshot,
            scenario=scenario,
            action_space_doc=action_space_doc,
            verdict_instructions=verdict_instructions,
        )
        try:
            data = await self.client.chat_json(
                system=self.persona.system_prompt,
                user=user_prompt,
                temperature=0.85,
                max_tokens=900,
            )
            return self._verdict_from_response(data, raw=str(data))
        except Exception as e:
            # Fail-soft: this persona abstains, swarm continues with others.
            return Verdict(
                persona_id=self.persona.id,
                persona_name=self.persona.name,
                role=self.persona.role,
                action_vector=[0.0] * len(ACTION_NAMES),
                reasoning="(abstained — call failed)",
                confidence=0.0,
                raw_response="",
                error=str(e),
            )

    # ------------------------- internals -------------------------
    def _build_user_prompt(
        self,
        state_snapshot: dict[str, Any],
        scenario: str,
        action_space_doc: str,
        verdict_instructions: str,
    ) -> str:
        state_lines = "\n".join(
            f"  - {k}: {self._fmt(v)}" for k, v in state_snapshot.items()
        )
        return (
            f"# Scenario\n{scenario}\n\n"
            f"# Current System State\n{state_lines}\n\n"
            f"# Available Interventions (action_vector — values in [0,1])\n"
            f"{action_space_doc}\n\n"
            f"# Your Task\n"
            f"From your perspective as {self.persona.name}, decide how strongly to "
            f"recommend each intervention right now. Speak in your own voice — your "
            f"lived experience and concerns must show through.\n\n"
            f"# Output Format\n{verdict_instructions}"
        )

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, list):
            return f"[{', '.join(SwarmAgent._fmt(x) for x in v[:6])}{'...' if len(v) > 6 else ''}]"
        return str(v)

    def _verdict_from_response(self, data: dict[str, Any], raw: str) -> Verdict:
        # Accept either dict-by-name or list form for the action vector.
        action_block = data.get("action_vector", {})
        vec: list[float] = []
        if isinstance(action_block, dict):
            for name in ACTION_NAMES:
                vec.append(self._coerce_unit(action_block.get(name, 0.0)))
        elif isinstance(action_block, list):
            for i in range(len(ACTION_NAMES)):
                vec.append(self._coerce_unit(action_block[i] if i < len(action_block) else 0.0))
        else:
            vec = [0.0] * len(ACTION_NAMES)

        confidence = self._coerce_unit(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", "")).strip()
        return Verdict(
            persona_id=self.persona.id,
            persona_name=self.persona.name,
            role=self.persona.role,
            action_vector=vec,
            reasoning=reasoning,
            confidence=confidence,
            raw_response=raw,
            error=None,
        )

    @staticmethod
    def _coerce_unit(x: Any) -> float:
        try:
            f = float(x)
        except (TypeError, ValueError):
            return 0.0
        if f < 0.0:
            return 0.0
        if f > 1.0:
            # Some models return 0..100 or 0..10 — normalize generously.
            if f <= 10.0:
                return f / 10.0
            if f <= 100.0:
                return f / 100.0
            return 1.0
        return f
