"""
Administrator: allocates resources under budget constraints.
Faces political risk — may under-implement costly policies.
"""

from .base_agent import BaseAgent
from env.state import SystemState
from typing import Dict

class AdminAgent(BaseAgent):

    def step(self, state: SystemState, incentives: Dict[str, float]) -> None:
        cost = incentives.get("total_cost", 0)

        # Deduct intervention costs from budget
        state.budget_remaining -= cost
        state.budget_utilization = self._clamp(
            1.0 - state.budget_remaining / 2_000_000.0
        )

        # Resource allocation quality depends on funding and transparency
        funding_effect = incentives.get("funding_boost", 0) * 0.15
        transparency_effect = incentives.get("transparency_report", 0) * 0.05
        corruption_noise = self.rng.uniform(-0.03, 0.01)  # admins may divert funds

        state.resource_allocation = self._clamp(
            state.resource_allocation + funding_effect + transparency_effect + corruption_noise
        )

        state.policy_compliance = self._clamp(
            state.policy_compliance
            + incentives.get("attendance_mandate", 0) * 0.05
            + incentives.get("transparency_report", 0) * 0.08
            - 0.01  # natural decay
            + self.rng.normal(0, 0.01)
        )