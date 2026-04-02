"""
Student agent: decides whether to attend, engage, or drop out.

Decision model: utility = f(teaching_quality, incentives, peer_norm, burnout)
"""

from .base_agent import BaseAgent
from env.state import SystemState
from typing import Dict

class StudentAgent(BaseAgent):

    def step(self, state: SystemState, incentives: Dict[str, float]) -> None:
        """Update dropout rate, attendance, and engagement."""

        # Utility signals
        teaching_quality = state.teacher_retention * (1 - state.teacher_burnout)
        scholarship_pull = incentives.get("student_scholarship", 0) * 0.15
        counseling_boost = incentives.get("counseling_programs", 0) * 0.08
        peer_norm = state.attendance_rate   # social conformity effect
        mandate_effect = incentives.get("attendance_mandate", 0) * 0.10

        # Marginal probability of staying in school
        p_stay = (
            0.30 * teaching_quality
            + 0.25 * peer_norm
            + 0.20 * scholarship_pull
            + 0.15 * counseling_boost
            + 0.10 * mandate_effect
        )
        p_stay = self._clamp(p_stay + self.rng.normal(0, 0.02))

        # Update state
        state.attendance_rate = self._clamp(
            state.attendance_rate * 0.95 + p_stay * 0.05
        )
        state.dropout_rate = self._clamp(
            state.dropout_rate + (1 - p_stay) * 0.02 - p_stay * 0.015
        )
        state.enrollment_rate = self._clamp(
            state.enrollment_rate - state.dropout_rate * 0.01
        )
        state.student_engagement = self._clamp(
            0.7 * state.student_engagement + 0.3 * (p_stay * teaching_quality)
        )