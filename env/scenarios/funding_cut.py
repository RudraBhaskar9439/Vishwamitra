"""
Scenario: Sudden government funding cut mid-episode.

Difficulty: easy
Shock: 25–40% budget cut at a random step between 15–25.
Initial state: moderately healthy system.
"""

import numpy as np
from .base_scenario import BaseScenario
from env.state import SystemState


class FundingCutScenario(BaseScenario):

    @property
    def difficulty(self) -> str:
        return "easy"

    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        s.enrollment_rate       = float(rng.uniform(0.75, 0.92))
        s.attendance_rate       = float(rng.uniform(0.70, 0.88))
        s.dropout_rate          = float(rng.uniform(0.06, 0.16))
        s.teacher_retention     = float(rng.uniform(0.72, 0.90))
        s.budget_utilization    = float(rng.uniform(0.55, 0.75))
        s.avg_class_size        = float(rng.uniform(28, 40))
        s.teacher_workload      = float(rng.uniform(0.55, 0.72))
        s.resource_allocation   = float(rng.uniform(0.55, 0.75))
        s.student_engagement    = float(rng.uniform(0.55, 0.78))
        s.teacher_burnout       = float(rng.uniform(0.18, 0.38))
        s.policy_compliance     = float(rng.uniform(0.65, 0.82))
        s.budget_remaining      = float(rng.uniform(800_000, 1_200_000))
        # Store shock timing in the state (custom field via __dict__)
        self._shock_step        = int(rng.integers(15, 26))
        self._shock_fired       = False
        return s

    def apply_shock(self, state: SystemState, step: int) -> None:
        if not self._shock_fired and step == self._shock_step:
            cut_pct = np.random.uniform(0.25, 0.40)
            state.budget_remaining *= (1.0 - cut_pct)
            self._shock_fired = True