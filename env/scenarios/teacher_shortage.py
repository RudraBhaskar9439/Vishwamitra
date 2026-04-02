"""
Scenario: Cascading teacher exodus — starts with low retention,
accelerates if the meta-agent doesn't intervene early.

Difficulty: medium
Shock: periodic teacher-leaving events every 10 steps.
Initial state: stressed system with high burnout.
"""

import numpy as np
from .base_scenario import BaseScenario
from env.state import SystemState


class TeacherShortageScenario(BaseScenario):

    @property
    def difficulty(self) -> str:
        return "medium"

    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        s.enrollment_rate       = float(rng.uniform(0.65, 0.82))
        s.attendance_rate       = float(rng.uniform(0.60, 0.78))
        s.dropout_rate          = float(rng.uniform(0.12, 0.25))
        s.teacher_retention     = float(rng.uniform(0.50, 0.68))  # starts low
        s.budget_utilization    = float(rng.uniform(0.70, 0.88))
        s.avg_class_size        = float(rng.uniform(38, 52))       # overcrowded
        s.teacher_workload      = float(rng.uniform(0.75, 0.92))   # overloaded
        s.resource_allocation   = float(rng.uniform(0.40, 0.60))
        s.student_engagement    = float(rng.uniform(0.40, 0.62))
        s.teacher_burnout       = float(rng.uniform(0.45, 0.68))   # high burnout
        s.policy_compliance     = float(rng.uniform(0.50, 0.70))
        s.budget_remaining      = float(rng.uniform(600_000, 900_000))
        self._last_shock_step   = 0
        return s

    def apply_shock(self, state: SystemState, step: int) -> None:
        # Every 10 steps: another wave of teachers leaves
        if step > 0 and step % 10 == 0:
            leave_fraction = np.random.uniform(0.04, 0.10)
            state.teacher_retention = max(
                0.10,
                state.teacher_retention - leave_fraction
            )
            # Larger classes as a downstream effect
            state.avg_class_size = min(65, state.avg_class_size + np.random.uniform(1, 3))