"""
Teacher agent: decides whether to stay, exert effort, or leave.

Burnout dynamics follow a logistic accumulation model.
"""

from .base_agent import BaseAgent
from env.state import SystemState
from typing import Dict

class TeacherAgent(BaseAgent):

    def step(self, state: SystemState, incentives: Dict[str, float]) -> None:
        # Stressors
        class_pressure = (state.avg_class_size - 25) / 35.0
        workload_stress = state.teacher_workload
        burnout_delta = 0.03 * class_pressure + 0.02 * workload_stress

        # Relief from incentives
        salary_relief = incentives.get("teacher_incentive", 0) * 0.12
        hire_relief = incentives.get("staff_hiring", 0) * 0.08   # more staff = less load

        state.teacher_burnout = self._clamp(
            state.teacher_burnout + burnout_delta - salary_relief - hire_relief
            + self.rng.normal(0, 0.01)
        )

        # Retention probability
        p_stay = self._clamp(
            0.90
            - state.teacher_burnout * 0.40
            + incentives.get("teacher_incentive", 0) * 0.20
            + self.rng.normal(0, 0.02)
        )

        state.teacher_retention = self._clamp(
            state.teacher_retention * 0.97 + p_stay * 0.03
        )

        # Class size grows when retention falls (fewer teachers → larger classes)
        state.avg_class_size = max(
            15,
            state.avg_class_size + (1 - state.teacher_retention) * 2.0
            - incentives.get("staff_hiring", 0) * 3.0,
        )
        state.teacher_workload = self._clamp(
            state.avg_class_size / 60.0 + state.teacher_burnout * 0.1
        )