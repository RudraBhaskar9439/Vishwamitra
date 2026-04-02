"""
VIDYA Scenario: Pandemic Recovery

Simulates post-COVID educational recovery with:
- Initial enrollment shock (learning loss)
- Gradual recovery phases
- Teacher burnout from remote teaching
- Infrastructure adaptation challenges
"""

import numpy as np
from .base_scenario import BaseScenario
from env.state import SystemState


class PandemicRecoveryScenario(BaseScenario):
    """
    Pandemic recovery scenario with phased reopening.
    
    Difficulty: medium
    Initial shock: 15-25% enrollment drop
    Recovery: Gradual over 60+ steps with setbacks
    """
    
    @property
    def difficulty(self) -> str:
        return "medium"
    
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        
        # Start with pandemic shock already applied
        enrollment_shock = rng.uniform(0.15, 0.25)
        
        s.enrollment_rate = float(rng.uniform(0.70, 0.85) - enrollment_shock)
        s.attendance_rate = float(rng.uniform(0.60, 0.75))  # Hybrid learning issues
        s.dropout_rate = float(rng.uniform(0.12, 0.22))  # Higher due to crisis
        s.teacher_retention = float(rng.uniform(0.65, 0.80))  # Burnout from remote
        s.budget_utilization = float(rng.uniform(0.70, 0.85))  # Emergency spending
        s.avg_class_size = float(rng.uniform(15, 25))  # Social distancing
        s.teacher_workload = float(rng.uniform(0.75, 0.90))  # Extra work for hybrid
        s.resource_allocation = float(rng.uniform(0.50, 0.70))  # Resources stretched
        s.student_engagement = float(rng.uniform(0.40, 0.60))  # Remote learning fatigue
        s.teacher_burnout = float(rng.uniform(0.35, 0.55))  # High burnout
        s.policy_compliance = float(rng.uniform(0.60, 0.75))  # Changing regulations
        s.budget_remaining = float(rng.uniform(600_000, 900_000))  # Depleted reserves
        
        # Recovery tracking
        self._recovery_phase = 0
        self._recovery_phases = [
            {'name': 'emergency', 'steps': (0, 15), 'budget_support': 1.2},
            {'name': 'hybrid', 'steps': (15, 40), 'budget_support': 1.1},
            {'name': 'reopening', 'steps': (40, 70), 'budget_support': 1.0},
            {'name': 'new_normal', 'steps': (70, 100), 'budget_support': 0.9}
        ]
        
        return s
    
    def apply_shock(self, state: SystemState, step: int) -> None:
        """Apply pandemic recovery dynamics."""
        # Determine current phase
        current_phase = None
        for phase in self._recovery_phases:
            start, end = phase['steps']
            if start <= step < end:
                current_phase = phase
                break
        
        if current_phase is None:
            return
        
        # Phase-specific dynamics
        if current_phase['name'] == 'emergency':
            # High stress, slow recovery
            state.teacher_burnout += 0.01
            state.student_engagement = max(0.3, state.student_engagement - 0.005)
            
            # Possible second wave shock
            if step == 10 and np.random.random() < 0.3:
                state.enrollment_rate *= 0.95
                state.budget_remaining *= 0.9
                
        elif current_phase['name'] == 'hybrid':
            # Gradual improvement
            state.student_engagement = min(0.8, state.student_engagement + 0.008)
            state.attendance_rate = min(0.9, state.attendance_rate + 0.005)
            
            # Teacher burnout slowly recovers
            state.teacher_burnout = max(0.3, state.teacher_burnout - 0.003)
            
        elif current_phase['name'] == 'reopening':
            # Faster recovery
            state.enrollment_rate = min(0.95, state.enrollment_rate + 0.01)
            state.dropout_rate = max(0.05, state.dropout_rate - 0.005)
            state.student_engagement = min(0.85, state.student_engagement + 0.01)
            
            # Catch-up learning pressure
            state.teacher_workload = min(0.95, state.teacher_workload + 0.01)
            
        elif current_phase['name'] == 'new_normal':
            # Stabilization
            state.teacher_retention = min(0.95, state.teacher_retention + 0.005)
            state.teacher_burnout = max(0.15, state.teacher_burnout - 0.002)
            
        # Budget pressure throughout
        if step % 5 == 0:
            state.budget_remaining -= np.random.uniform(20000, 50000)


class LearningLossScenario(BaseScenario):
    """
    Focus on addressing pandemic learning loss.
    
    Difficulty: hard
    Challenge: Students far behind, accelerated catch-up needed
    """
    
    @property
    def difficulty(self) -> str:
        return "hard"
    
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        
        # Severe learning loss
        s.enrollment_rate = float(rng.uniform(0.75, 0.85))
        s.attendance_rate = float(rng.uniform(0.65, 0.80))
        s.dropout_rate = float(rng.uniform(0.15, 0.30))  # High risk students
        s.teacher_retention = float(rng.uniform(0.60, 0.75))  # Exhausted teachers
        s.budget_utilization = float(rng.uniform(0.80, 0.95))  # Maxed out
        s.avg_class_size = float(rng.uniform(20, 30))  # Smaller for catch-up
        s.teacher_workload = float(rng.uniform(0.85, 0.98))  # Nearly overwhelmed
        s.resource_allocation = float(rng.uniform(0.60, 0.75))
        s.student_engagement = float(rng.uniform(0.35, 0.50))  # Demotivated
        s.teacher_burnout = float(rng.uniform(0.45, 0.65))
        s.policy_compliance = float(rng.uniform(0.70, 0.85))
        s.budget_remaining = float(rng.uniform(400_000, 700_000))
        
        # Learning gap metric
        self._learning_gap = rng.uniform(0.3, 0.5)  # 30-50% behind
        self._catch_up_progress = 0.0
        
        return s
    
    def apply_shock(self, state: SystemState, step: int) -> None:
        """Apply learning loss recovery dynamics."""
        # Without intervention, students fall further behind
        if state.student_engagement < 0.5:
            self._learning_gap += 0.002
        else:
            # Slow catch-up with good engagement
            progress = 0.01 * (state.student_engagement - 0.4)
            self._learning_gap = max(0, self._learning_gap - progress)
            self._catch_up_progress += progress
        
        # High-intensity interventions drain budget faster
        intervention_intensity = (
            state.budget_utilization + 
            state.teacher_workload + 
            (1 - state.student_engagement)
        ) / 3
        
        if intervention_intensity > 0.8 and step % 3 == 0:
            state.teacher_burnout += 0.005  # Teacher exhaustion
            
        # Parental pressure increases over time
        if step > 30 and step % 10 == 0:
            # External pressure shock
            state.policy_compliance *= 0.95  # Harder to satisfy everyone


class HybridLearningScenario(BaseScenario):
    """
    Ongoing hybrid learning challenges.
    
    Difficulty: medium
    Challenge: Managing simultaneous in-person and remote instruction
    """
    
    @property
    def difficulty(self) -> str:
        return "medium"
    
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        
        s.enrollment_rate = float(rng.uniform(0.78, 0.90))
        s.attendance_rate = float(rng.uniform(0.55, 0.70))  # Split between modes
        s.dropout_rate = float(rng.uniform(0.10, 0.18))
        s.teacher_retention = float(rng.uniform(0.68, 0.82))
        s.budget_utilization = float(rng.uniform(0.75, 0.88))
        s.avg_class_size = float(rng.uniform(12, 20))  # Half capacity
        s.teacher_workload = float(rng.uniform(0.80, 0.95))  # Double prep
        s.resource_allocation = float(rng.uniform(0.65, 0.80))
        s.student_engagement = float(rng.uniform(0.45, 0.65))
        s.teacher_burnout = float(rng.uniform(0.40, 0.58))
        s.policy_compliance = float(rng.uniform(0.70, 0.85))
        s.budget_remaining = float(rng.uniform(500_000, 800_000))
        
        # Hybrid tracking
        self._remote_students = rng.uniform(0.3, 0.5)
        self._tech_issues = 0.0
        
        return s
    
    def apply_shock(self, state: SystemState, step: int) -> None:
        """Apply hybrid learning dynamics."""
        # Technology issues randomly spike
        if np.random.random() < 0.1:  # 10% chance per step
            self._tech_issues += np.random.uniform(0.05, 0.15)
            state.student_engagement *= 0.98
            state.teacher_burnout += 0.008
        else:
            self._tech_issues = max(0, self._tech_issues - 0.02)
        
        # Equity concerns grow if remote students neglected
        if state.resource_allocation < 0.6 and self._remote_students > 0.3:
            state.enrollment_rate *= 0.995
            state.dropout_rate += 0.002
            
        # Periodic internet/connectivity crises
        if step % 15 == 0 and np.random.random() < 0.3:
            state.attendance_rate *= 0.90
            state.teacher_workload += 0.05
