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
    ):
        self.config_path = Path(config_path)
        self.prompts_dir = Path(prompts_dir)
        self.client = client or LLMClient()
        self.dissonance_threshold = dissonance_threshold

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
    ) -> ResonanceReport:
        """Run all swarms in parallel and aggregate."""
        swarm_verdicts = await asyncio.gather(
            *[
                s.deliberate(
                    state_snapshot=state_snapshot,
                    scenario=scenario,
                    action_space_doc=self._action_space_doc,
                    verdict_instructions=self._verdict_instructions,
                )
                for s in self.swarms
            ]
        )
        report = compute_resonance(
            swarm_verdicts=list(swarm_verdicts),
            dissonance_threshold=self.dissonance_threshold,
            scenario=scenario,
            state_snapshot=state_snapshot,
        )
        if self.logger is not None:
            self.logger.log_report(report)
        return report

    def deliberate_sync(
        self,
        state_snapshot: dict[str, Any],
        scenario: str = "general",
    ) -> ResonanceReport:
        """Sync convenience wrapper for non-async callers (CLI / scripts)."""
        return asyncio.run(self.deliberate(state_snapshot, scenario))

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
        }
