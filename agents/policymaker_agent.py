"""
Policymaker: introduces delayed systemic shocks or boosts.
Models political short-termism.
"""

from .base_agent import BaseAgent
from env.state import SystemState
from typing import Dict

class PolicymakerAgent(BaseAgent):

    def __init__(self, state, rng):
        super().__init__(state, rng)
        self._next_shock_at = int(rng.integers(15, 30))

    def step(self, state: SystemState, incentives: Dict[str, float]) -> None:
        # Policymakers respond to transparency with better funding
        if incentives.get("transparency_report", 0) > 0.5:
            state.budget_remaining += 50_000 * incentives["transparency_report"]

        # Delayed budget shock (simulates election-cycle funding cuts)
        if state.step == self._next_shock_at:
            cut = self.rng.uniform(0.10, 0.30)
            state.budget_remaining *= (1 - cut)
            self._next_shock_at = state.step + int(self.rng.integers(20, 40))