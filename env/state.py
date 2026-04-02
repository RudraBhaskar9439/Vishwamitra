from dataclasses import dataclass, field
import resource
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class SystemState:
    # Core health metrics 
    enrollment_rate: float = 0.85
    attendance_rate: float = 0.80
    dropout_rate: float = 0.10
    teacher_retention: float = 0.85
    budget_utilization: float = 0.75

    # Operational metrics
    avg_class_size: float = 35.0  # students per class
    teacher_workload: float = 0.70 
    resource_allocation: float = 0.65  # quality of resource dist

    # Behavioral signals
    student_engagement: float = 0.65
    teacher_burnout: float = 0.30  # lower is better
    policy_compliance: float = 0.75  # higher is better

    # Financial
    budget_remaining: float = 1_000_000.0  # absolute currency units
    step: int = 0

    # History for reward smoothing
    health_history: list = field(default_factory=list)

    def to_obs_array(self) -> np.ndarray:
        """Returns flat observation vector for RL agent."""
        return np.array([
            self.enrollment_rate,
            self.attendance_rate,
            self.dropout_rate,
            self.teacher_retention,
            self.budget_utilization,
            self.avg_class_size / 60.0,
            self.teacher_workload,
            self.resource_allocation,
            self.student_engagement,
            self.teacher_burnout,
            self.policy_compliance,
            self.budget_remaining / 2_000_000.0,
            self.step,
        ], dtype=np.float32)

    @property
    def health_score(self) -> float:
        return (
            0.25 * self.enrollment_rate +
            0.20 * self.attendance_rate +
            0.20 * self.teacher_retention +
            0.15 * self.student_engagement +
            0.10 * self.resource_allocation +
            0.10 * (1.0 - self.teacher_burnout)
            - 0.30 * self.dropout_rate
        )
    
