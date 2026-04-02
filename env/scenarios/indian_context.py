"""
Scenario: Indian education system context.

Models:
- Ghost enrollment (fake students registered for funding)
- Seasonal migration affecting attendance
- Rural school consolidation effects
- Private school migration pull

Difficulty: hard
Initial state: high enrollment on paper, low actual attendance.
"""

import numpy as np
from .base_scenario import BaseScenario
from env.state import SystemState


class IndianContextScenario(BaseScenario):

    @property
    def difficulty(self) -> str:
        return "hard"

    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        # Ghost enrollment: enrollment_rate looks high but attendance is low
        s.enrollment_rate       = float(rng.uniform(0.85, 0.97))   # inflated
        s.attendance_rate       = float(rng.uniform(0.40, 0.62))   # real is low
        s.dropout_rate          = float(rng.uniform(0.18, 0.35))   # high true dropout
        s.teacher_retention     = float(rng.uniform(0.60, 0.78))
        s.budget_utilization    = float(rng.uniform(0.80, 0.95))   # budget stretched
        s.avg_class_size        = float(rng.uniform(42, 60))
        s.teacher_workload      = float(rng.uniform(0.72, 0.88))
        s.resource_allocation   = float(rng.uniform(0.35, 0.55))   # poor distribution
        s.student_engagement    = float(rng.uniform(0.35, 0.55))
        s.teacher_burnout       = float(rng.uniform(0.35, 0.58))
        s.policy_compliance     = float(rng.uniform(0.35, 0.58))   # low compliance
        s.budget_remaining      = float(rng.uniform(500_000, 800_000))

        # Seasonal migration shock schedule
        # Steps 20–30: harvest season — rural students absent
        # Steps 50–55: exam pressure drop
        self._migration_window  = (int(rng.integers(18, 25)), int(rng.integers(28, 35)))
        self._private_pull_step = int(rng.integers(30, 50))
        self._private_pull_fired = False
        return s

    def apply_shock(self, state: SystemState, step: int) -> None:
        # Seasonal migration: attendance drops sharply
        mig_start, mig_end = self._migration_window
        if mig_start <= step <= mig_end:
            migration_drag = 0.015
            state.attendance_rate = max(0.10, state.attendance_rate - migration_drag)
            state.student_engagement = max(0.05, state.student_engagement - 0.01)

        # Private school pull: families with means migrate to private
        if not self._private_pull_fired and step == self._private_pull_step:
            pull = np.random.uniform(0.05, 0.12)
            state.enrollment_rate = max(0.30, state.enrollment_rate - pull)
            state.dropout_rate = min(0.50, state.dropout_rate + pull * 0.5)
            self._private_pull_fired = True

        # Ghost enrollment correction: transparency reports reduce fake numbers
        # (handled via policy_compliance in admin_agent, but add noise here)
        if step % 15 == 0:
            fraud_correction = np.random.uniform(-0.03, 0.01)
            state.enrollment_rate = max(0.20, state.enrollment_rate + fraud_correction)