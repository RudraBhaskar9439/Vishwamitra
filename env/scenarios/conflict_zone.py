"""
VIDYA Scenario: Conflict Zone

Simulates education system under protracted conflict:
- Ongoing security threats
- Teacher flight and infrastructure damage
- Student trauma and displacement
- International aid dependency
"""

import numpy as np
from .base_scenario import BaseScenario
from env.state import SystemState


class ConflictZoneScenario(BaseScenario):
    """
    Protracted conflict scenario.
    
    Difficulty: hard
    Continuous stress with periodic acute crises
    """
    
    @property
    def difficulty(self) -> str:
        return "hard"
    
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        
        # Already degraded by ongoing conflict
        s.enrollment_rate = float(rng.uniform(0.55, 0.75))
        s.attendance_rate = float(rng.uniform(0.50, 0.70))  # Security issues
        s.dropout_rate = float(rng.uniform(0.20, 0.40))  # High
        s.teacher_retention = float(rng.uniform(0.40, 0.60))  # Massive flight
        s.budget_utilization = float(rng.uniform(0.40, 0.60))  # Disrupted
        s.avg_class_size = float(rng.uniform(35, 55))  # Overcrowded (displaced)
        s.teacher_workload = float(rng.uniform(0.70, 0.85))
        s.resource_allocation = float(rng.uniform(0.30, 0.50))  # Damaged infrastructure
        s.student_engagement = float(rng.uniform(0.30, 0.50))  # Trauma
        s.teacher_burnout = float(rng.uniform(0.50, 0.70))  # Danger pay stress
        s.policy_compliance = float(rng.uniform(0.40, 0.60))  
        s.budget_remaining = float(rng.uniform(200_000, 500_000))  # Limited
        
        # Conflict dynamics
        self._conflict_intensity = rng.uniform(0.4, 0.7)
        self._infrastructure_damage = rng.uniform(0.2, 0.5)
        self._displaced_students = rng.uniform(0.15, 0.35)
        self._aid_dependency = rng.uniform(0.3, 0.6)
        
        return s
    
    def apply_shock(self, state: SystemState, step: int) -> None:
        """Apply conflict zone dynamics."""
        # Continuous low-level deterioration
        if step % 3 == 0:
            state.teacher_retention = max(0.2, state.teacher_retention - 0.005)
            state.infrastructure_health = max(0.1, 0.5 - self._infrastructure_damage * 0.5)
        
        # Random acute incidents (attacks, displacement waves)
        if np.random.random() < self._conflict_intensity * 0.05:
            # Acute crisis
            severity = np.random.uniform(0.1, 0.4)
            
            state.enrollment_rate *= (1 - severity * 0.3)
            state.teacher_retention *= (1 - severity * 0.4)  # Teachers flee
            self._displaced_students += severity * 0.1
            self._infrastructure_damage = min(0.9, self._infrastructure_damage + severity * 0.2)
            
            # Budget shock (emergency response)
            state.budget_remaining *= (1 - severity * 0.2)
            
        # International aid arrives periodically
        if step % 20 == 0 and step > 0:
            aid_amount = np.random.uniform(100000, 300000) * self._aid_dependency
            state.budget_remaining += aid_amount
            state.teacher_retention = min(0.95, state.teacher_retention + 0.05)  # Retention incentives
            
        # Student trauma affects engagement
        state.student_engagement = max(0.1, 0.6 - self._conflict_intensity * 0.4)
        
        # Overcrowding from displacement
        if self._displaced_students > 0.3:
            state.avg_class_size = min(60, 40 + self._displaced_students * 20)
            state.teacher_workload = min(0.95, state.teacher_workload + 0.01)


class NaturalDisasterScenario(BaseScenario):
    """
    Post-natural disaster recovery.
    
    Difficulty: medium-hard
    Sudden infrastructure loss, gradual rebuilding
    """
    
    @property
    def difficulty(self) -> str:
        return "hard"
    
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        
        # Immediate post-disaster
        disaster_impact = rng.uniform(0.3, 0.6)
        
        s.enrollment_rate = float(rng.uniform(0.65, 0.80) - disaster_impact * 0.2)
        s.attendance_rate = float(rng.uniform(0.60, 0.75) - disaster_impact * 0.15)
        s.dropout_rate = float(rng.uniform(0.12, 0.20) + disaster_impact * 0.1)
        s.teacher_retention = float(rng.uniform(0.65, 0.80) - disaster_impact * 0.2)
        s.budget_utilization = float(rng.uniform(0.70, 0.85))
        s.avg_class_size = float(rng.uniform(30, 45))  # Displaced students
        s.teacher_workload = float(rng.uniform(0.75, 0.90))  # Stress + displacement
        s.resource_allocation = float(rng.uniform(0.40, 0.60))  # Damaged
        s.student_engagement = float(rng.uniform(0.50, 0.65) - disaster_impact * 0.1)
        s.teacher_burnout = float(rng.uniform(0.40, 0.55))
        s.policy_compliance = float(rng.uniform(0.70, 0.85))
        s.budget_remaining = float(rng.uniform(500_000, 800_000))
        
        # Recovery parameters
        self._disaster_severity = disaster_impact
        self._rebuilding_progress = 0.0
        self._emergency_mode = True
        
        return s
    
    def apply_shock(self, state: SystemState, step: int) -> None:
        """Apply disaster recovery dynamics."""
        if self._emergency_mode and step < 15:
            # Emergency phase: stabilization
            if step % 3 == 0:
                # Emergency spending
                state.budget_remaining -= np.random.uniform(30000, 60000)
                state.teacher_burnout += 0.008
                
        elif step >= 15:
            # Rebuilding phase
            self._emergency_mode = False
            self._rebuilding_progress = min(1.0, (step - 15) / 60)
            
            # Gradual improvement
            recovery_factor = self._rebuilding_progress * 0.3
            
            state.enrollment_rate = min(0.95, state.enrollment_rate + 0.003)
            state.teacher_retention = min(0.90, 0.60 + recovery_factor)
            state.resource_allocation = min(0.90, 0.50 + recovery_factor)
            state.student_engagement = min(0.85, 0.55 + recovery_factor)
            
            # Rebuilding costs
            if step % 5 == 0:
                state.budget_remaining -= np.random.uniform(40000, 80000)
                
        # Setbacks possible
        if step > 20 and step % 25 == 0 and np.random.random() < 0.3:
            # Rebuilding setback (funding shortfall, new damage)
            state.budget_remaining *= 0.85
            state.resource_allocation *= 0.90


class DisplacementCrisisScenario(BaseScenario):
    """
    Large-scale refugee/displacement crisis.
    
    Difficulty: hard
    Challenge: Absorbing large influx of displaced students
    """
    
    @property
    def difficulty(self) -> str:
        return "hard"
    
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        s = SystemState()
        
        # System already strained
        s.enrollment_rate = float(rng.uniform(0.70, 0.85))
        s.attendance_rate = float(rng.uniform(0.70, 0.85))
        s.dropout_rate = float(rng.uniform(0.10, 0.18))
        s.teacher_retention = float(rng.uniform(0.70, 0.85))
        s.budget_utilization = float(rng.uniform(0.65, 0.80))
        s.avg_class_size = float(rng.uniform(25, 35))
        s.teacher_workload = float(rng.uniform(0.65, 0.80))
        s.resource_allocation = float(rng.uniform(0.60, 0.75))
        s.student_engagement = float(rng.uniform(0.60, 0.75))
        s.teacher_burnout = float(rng.uniform(0.30, 0.45))
        s.policy_compliance = float(rng.uniform(0.75, 0.88))
        s.budget_remaining = float(rng.uniform(700_000, 1_000_000))
        
        # Displacement wave parameters
        self._displacement_wave_step = int(rng.integers(10, 25))
        self._displaced_ratio = rng.uniform(0.3, 0.6)  # 30-60% population increase
        self._wave_arrived = False
        
        return s
    
    def apply_shock(self, state: SystemState, step: int) -> None:
        """Apply displacement dynamics."""
        # Displacement wave arrival
        if not self._wave_arrived and step == self._displacement_wave_step:
            self._wave_arrived = True
            
            # Sudden system shock
            enrollment_increase = self._displaced_ratio
            state.enrollment_rate = min(0.98, state.enrollment_rate * (1 + enrollment_increase * 0.3))
            state.avg_class_size = min(70, state.avg_class_size * (1 + enrollment_increase))
            state.teacher_workload = min(0.95, state.teacher_workload + 0.15)
            state.resource_allocation *= 0.7
            
            # Budget emergency
            state.budget_remaining *= 0.8
            
        if self._wave_arrived:
            # Ongoing integration challenges
            if step % 5 == 0:
                # Language/cultural integration costs
                state.budget_remaining -= np.random.uniform(20000, 40000)
                
            # Social cohesion strain
            if state.resource_allocation < 0.5:
                state.student_engagement = max(0.3, state.student_engagement - 0.005)
                state.dropout_rate = min(0.5, state.dropout_rate + 0.003)
                
            # Teacher strain from diverse classroom needs
            if state.avg_class_size > 50:
                state.teacher_burnout += 0.005
                state.teacher_retention *= 0.995
