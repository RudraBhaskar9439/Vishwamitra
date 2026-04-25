from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError("pyyaml required: pip install pyyaml") from e

try:
    from dotenv import load_dotenv
    load_dotenv()
    # Also try repo-level .env locations.
    for candidate in [
        Path(__file__).resolve().parents[2] / "spaces" / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate, override=False)
except ImportError:
    pass

from swarms.core.persona import Persona
from swarms.core.swarm import Swarm
from swarms.core.llm_client import LLMClient
from swarms.core.verdict import ResonanceReport
from swarms.orchestrator.resonance import compute_resonance
from swarms.orchestrator.round_log import RoundLogger
from swarms.orchestrator.router import (
    WeightAllocator, OrchestratorPlan, PersonaAllocator,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWARMS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = _SWARMS_ROOT / "config" / "roles.yaml"
DEFAULT_PROMPTS = _SWARMS_ROOT / "prompts"


class SwarmManager:
    """
    Top-level orchestrator. Loads roles+personas from YAML, builds one
    `Swarm` per role, deliberates them in parallel, computes resonance,
    and (optionally) writes a JSONL round log.

    Usage:
        manager = SwarmManager()
        report = await manager.deliberate(state_snapshot, scenario="funding_cut")
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        prompts_dir: str | Path = DEFAULT_PROMPTS,
        client: LLMClient | None = None,
        log: bool = True,
        run_id: str | None = None,
        dissonance_threshold: float = 0.55,
        router: WeightAllocator | None = None,
        persona_router: PersonaAllocator | None = None,
    ):
        self.config_path = Path(config_path)
        self.prompts_dir = Path(prompts_dir)
        self.client = client or LLMClient()
        self.dissonance_threshold = dissonance_threshold
        self.router = router or WeightAllocator()
        self.persona_router = persona_router or PersonaAllocator()

        self._action_space_doc = (self.prompts_dir / "action_space.txt").read_text(encoding="utf-8")
        self._verdict_instructions = (self.prompts_dir / "verdict_instructions.txt").read_text(encoding="utf-8")

        self.swarms: list[Swarm] = self._load_swarms()
        self.logger = RoundLogger(run_id=run_id) if log else None

    # ------------------------- loading -------------------------
    def _load_swarms(self) -> list[Swarm]:
        cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        roles = cfg.get("roles", {})
        if not roles:
            raise ValueError(f"No 'roles' found in {self.config_path}")

        swarms: list[Swarm] = []
        for role_name, role_cfg in roles.items():
            personas_cfg = role_cfg.get("personas", [])
            if not personas_cfg:
                continue
            personas = [
                Persona(
                    id=p["id"],
                    role=role_name,
                    name=p["name"],
                    system_prompt=p["system_prompt"].strip(),
                    traits=p.get("traits", {}),
                    fit_signals=p.get("fit_signals", {}),
                )
                for p in personas_cfg
            ]
            swarms.append(Swarm(role=role_name, personas=personas, client=self.client))
        return swarms

    # ------------------------- deliberation -------------------------
    async def deliberate(
        self,
        state_snapshot: dict[str, Any],
        scenario: str = "general",
        *,
        orchestrator_overrides: dict[str, Any] | None = None,
    ) -> ResonanceReport:
        """Run all swarms in parallel, routed by the orchestrator, and aggregate.

        `orchestrator_overrides` (optional) is forwarded to the WeightAllocator
        and lets a caller pin a specific (model, verdict_weight) per role —
        used by the UI's MANUAL mode.
        """
        plan: OrchestratorPlan = self.router.allocate(
            state=state_snapshot,
            manual_overrides=orchestrator_overrides,
        )

        # L2: compute per-persona fit weights for each swarm BEFORE
        # deliberation (cheap, no LLM calls). Each swarm gets its own
        # persona_id → weight mapping to use during aggregation.
        l2_per_swarm: dict[str, list] = {}
        for s in self.swarms:
            l2_per_swarm[s.role] = self.persona_router.allocate(
                personas=[a.persona for a in s.agents],
                state=state_snapshot,
            )

        swarm_verdicts = await asyncio.gather(
            *[
                s.deliberate(
                    state_snapshot=state_snapshot,
                    scenario=scenario,
                    action_space_doc=self._action_space_doc,
                    verdict_instructions=self._verdict_instructions,
                    model=plan.model_for(s.role),
                    persona_weights={
                        d.persona_id: d.weight for d in l2_per_swarm[s.role]
                    },
                    persona_fits=[d.to_dict() for d in l2_per_swarm[s.role]],
                )
                for s in self.swarms
            ]
        )

        swarm_weights = {role: plan.weight_for(role) for role in plan.decisions}
        report = compute_resonance(
            swarm_verdicts=list(swarm_verdicts),
            dissonance_threshold=self.dissonance_threshold,
            scenario=scenario,
            state_snapshot=state_snapshot,
            swarm_weights=swarm_weights,
        )
        report.orchestrator_plan = plan.to_dict()

        if self.logger is not None:
            self.logger.log_report(report)
        return report

    def deliberate_sync(
        self,
        state_snapshot: dict[str, Any],
        scenario: str = "general",
        *,
        orchestrator_overrides: dict[str, Any] | None = None,
    ) -> ResonanceReport:
        """Sync convenience wrapper for non-async callers (CLI / scripts)."""
        return asyncio.run(
            self.deliberate(
                state_snapshot, scenario,
                orchestrator_overrides=orchestrator_overrides,
            )
        )

    def preview_plan(
        self,
        state_snapshot: dict[str, Any],
        orchestrator_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return what the router WOULD allocate for this state without
        actually running any LLM calls. Used by the UI to preview the
        orchestrator's decisions."""
        return self.router.allocate(state_snapshot, orchestrator_overrides).to_dict()

    # ------------------------- introspection -------------------------
    @property
    def roles(self) -> list[str]:
        return [s.role for s in self.swarms]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.client.config.model,
            "provider": self.client.config.provider_name,
            "roles": {
                s.role: [a.persona.name for a in s.agents] for s in self.swarms
            },
            "orchestrator": {
                "heavy_model": self.router.heavy_model,
                "light_model": self.router.light_model,
                "attention_threshold": self.router.attention_threshold,
                "heavy_verdict_weight": self.router.heavy_verdict_weight,
                "light_verdict_weight": self.router.light_verdict_weight,
            },
        }
