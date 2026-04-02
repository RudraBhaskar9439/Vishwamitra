"""
VIDYA Adversarial Agent

Adversarial agent for stress-testing educational systems and
training robust intervention policies through worst-case scenarios.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from agents.base_agent import BaseAgent
from env.state import SystemState


class AdversarialAgent(BaseAgent):
    """
    Adversarial agent that tries to cause system collapse.
    
    Used for:
    - Stress testing intervention policies
    - Generating worst-case training scenarios
    - Identifying system vulnerabilities
    - Training robust RL agents
    """
    
    def __init__(
        self,
        state: SystemState,
        rng: np.random.Generator,
        attack_strategy: str = 'targeted',
        intensity: float = 0.5
    ):
        super().__init__(state, rng)
        self.attack_strategy = attack_strategy
        self.intensity = intensity
        self.attack_history = []
        
    def step(self, state: SystemState, incentives: Dict[str, float]) -> None:
        """
        Execute adversarial attack on the system.
        
        The adversary tries to maximize dropout and minimize
        system health through strategic interventions.
        """
        # Determine attack vector based on strategy
        if self.attack_strategy == 'random':
            self._random_attack(state)
        elif self.attack_strategy == 'targeted':
            self._targeted_attack(state)
        elif self.attack_strategy == 'coordinated':
            self._coordinated_attack(state, incentives)
        elif self.attack_strategy == 'adaptive':
            self._adaptive_attack(state)
        else:
            self._targeted_attack(state)
            
        # Log attack
        self.attack_history.append({
            'step': state.step,
            'strategy': self.attack_strategy,
            'dropout_before': state.dropout_rate,
        })
        
    def _random_attack(self, state: SystemState) -> None:
        """Random attacks on system parameters."""
        # Randomly degrade random metrics
        attack_strength = self.intensity * self.rng.uniform(0.5, 1.5)
        
        target = self.rng.choice([
            'enrollment', 'attendance', 'teacher_retention', 
            'engagement', 'budget'
        ])
        
        if target == 'enrollment':
            state.enrollment_rate *= (1 - attack_strength * 0.1)
        elif target == 'attendance':
            state.attendance_rate *= (1 - attack_strength * 0.15)
        elif target == 'teacher_retention':
            state.teacher_retention *= (1 - attack_strength * 0.2)
        elif target == 'engagement':
            state.student_engagement *= (1 - attack_strength * 0.2)
        elif target == 'budget':
            state.budget_remaining *= (1 - attack_strength * 0.25)
            
    def _targeted_attack(self, state: SystemState) -> None:
        """Target the most vulnerable system component."""
        attack_strength = self.intensity
        
        # Identify most vulnerable component
        vulnerabilities = {
            'enrollment': 1 - state.enrollment_rate,
            'attendance': 1 - state.attendance_rate,
            'teacher_retention': 1 - state.teacher_retention,
            'engagement': 1 - state.student_engagement,
            'budget': max(0, 1 - state.budget_remaining / 1e6),
        }
        
        # Target the weakest link
        target = max(vulnerabilities, key=vulnerabilities.get)
        
        # Amplify existing problems (positive feedback)
        if target == 'enrollment':
            state.dropout_rate += attack_strength * 0.02
            state.enrollment_rate -= attack_strength * 0.02
        elif target == 'attendance':
            state.attendance_rate -= attack_strength * 0.03
            state.student_engagement -= attack_strength * 0.01
        elif target == 'teacher_retention':
            state.teacher_retention -= attack_strength * 0.04
            state.teacher_burnout += attack_strength * 0.03
        elif target == 'engagement':
            state.student_engagement -= attack_strength * 0.05
            state.dropout_rate += attack_strength * 0.01
        elif target == 'budget':
            state.budget_remaining -= attack_strength * 100000
            
        # Ensure values stay in valid range
        self._clamp_state(state)
        
    def _coordinated_attack(self, state: SystemState, incentives: Dict[str, float]) -> None:
        """
        Coordinate attack with knowledge of defender's incentives.
        
        Attacks weak points that aren't being protected.
        """
        attack_strength = self.intensity
        
        # Find where defender is investing least
        incentive_values = {
            'funding_boost': incentives.get('funding_boost', 0),
            'teacher_incentive': incentives.get('teacher_incentive', 0),
            'student_scholarship': incentives.get('student_scholarship', 0),
            'staff_hiring': incentives.get('staff_hiring', 0),
            'counseling_programs': incentives.get('counseling_programs', 0),
        }
        
        # Attack the least protected area
        weakest_protection = min(incentive_values, key=incentive_values.get)
        
        # Map protection to system component
        attack_map = {
            'funding_boost': 'budget',
            'teacher_incentive': 'teacher_retention',
            'student_scholarship': 'enrollment',
            'staff_hiring': 'teacher_retention',
            'counseling_programs': 'engagement',
        }
        
        target = attack_map.get(weakest_protection, 'engagement')
        
        # Execute attack with extra strength due to weak defense
        if target == 'budget':
            state.budget_remaining -= attack_strength * 150000
        elif target == 'teacher_retention':
            state.teacher_retention -= attack_strength * 0.05
            state.teacher_burnout += attack_strength * 0.04
        elif target == 'enrollment':
            state.enrollment_rate -= attack_strength * 0.03
            state.dropout_rate += attack_strength * 0.02
        elif target == 'engagement':
            state.student_engagement -= attack_strength * 0.06
            
        self._clamp_state(state)
        
    def _adaptive_attack(self, state: SystemState) -> None:
        """
        Adapt attack based on system response.
        
        Learns which attacks are most effective and adjusts.
        """
        # Compute recent attack effectiveness
        if len(self.attack_history) >= 3:
            recent = self.attack_history[-3:]
            # If dropout not increasing, increase intensity
            if recent[-1]['dropout_before'] <= recent[0]['dropout_before']:
                self.intensity = min(1.0, self.intensity * 1.1)
            else:
                # Attack working, maintain or slightly decrease
                self.intensity = max(0.3, self.intensity * 0.95)
                
        # Execute targeted attack with adapted intensity
        self._targeted_attack(state)
        
    def _clamp_state(self, state: SystemState) -> None:
        """Ensure state values remain in valid ranges."""
        state.enrollment_rate = self._clamp(state.enrollment_rate, 0, 1)
        state.attendance_rate = self._clamp(state.attendance_rate, 0, 1)
        state.dropout_rate = self._clamp(state.dropout_rate, 0, 0.5)
        state.teacher_retention = self._clamp(state.teacher_retention, 0, 1)
        state.teacher_burnout = self._clamp(state.teacher_burnout, 0, 1)
        state.student_engagement = self._clamp(state.student_engagement, 0, 1)


class StressTestRunner:
    """
    Run comprehensive stress tests on intervention policies.
    """
    
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        
    def run_stress_test(
        self,
        n_episodes: int = 100,
        attack_strategies: Optional[list] = None,
        intensity_levels: Optional[list] = None
    ) -> Dict:
        """
        Run comprehensive stress test with varying attack strategies.
        
        Returns:
            Dict with collapse rates under different attack types
        """
        attack_strategies = attack_strategies or ['random', 'targeted', 'adaptive']
        intensity_levels = intensity_levels or [0.3, 0.5, 0.7, 0.9]
        
        results = {}
        
        for strategy in attack_strategies:
            results[strategy] = {}
            for intensity in intensity_levels:
                collapse_count = 0
                total_reward = 0
                
                for episode in range(n_episodes):
                    # Create adversarial environment
                    from env.dropout_env import DropoutCommonsEnv
                    env = DropoutCommonsEnv()
                    
                    # Inject adversarial agent
                    obs, info = env.reset(seed=episode)
                    env._adversarial_agent = AdversarialAgent(
                        env.state, env.np_random,
                        attack_strategy=strategy,
                        intensity=intensity
                    )
                    
                    episode_reward = 0
                    terminated = False
                    
                    while not terminated:
                        action = self.policy(obs)
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        # Adversary attacks after each step
                        env._adversarial_agent.step(env.state, {})
                        
                        episode_reward += reward
                        
                        if terminated or truncated:
                            break
                            
                    if terminated:  # Collapse occurred
                        collapse_count += 1
                    total_reward += episode_reward
                    
                results[strategy][intensity] = {
                    'collapse_rate': collapse_count / n_episodes,
                    'avg_reward': total_reward / n_episodes,
                    'robustness_score': 1 - (collapse_count / n_episodes),
                }
                
        return results
        
    def find_vulnerabilities(self, n_episodes: int = 50) -> Dict[str, float]:
        """
        Identify which system components are most vulnerable.
        
        Returns:
            Dict mapping component to vulnerability score (0-1)
        """
        vulnerabilities = {}
        
        components = [
            'enrollment', 'attendance', 'teacher_retention',
            'engagement', 'budget'
        ]
        
        for component in components:
            collapse_count = 0
            
            for episode in range(n_episodes):
                obs, info = self.env.reset(seed=episode)
                terminated = False
                
                while not terminated:
                    # Normal policy action
                    action = self.policy(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # Targeted attack on specific component
                    if component == 'enrollment':
                        self.env.state.enrollment_rate *= 0.98
                    elif component == 'attendance':
                        self.env.state.attendance_rate *= 0.97
                    elif component == 'teacher_retention':
                        self.env.state.teacher_retention *= 0.96
                    elif component == 'engagement':
                        self.env.state.student_engagement *= 0.97
                    elif component == 'budget':
                        self.env.state.budget_remaining *= 0.95
                        
                    if terminated or truncated:
                        break
                        
                if terminated:
                    collapse_count += 1
                    
            vulnerabilities[component] = collapse_count / n_episodes
            
        return vulnerabilities
        
    def generate_robustness_report(self) -> str:
        """Generate detailed robustness analysis report."""
        stress_results = self.run_stress_test(n_episodes=50)
        vulnerabilities = self.find_vulnerabilities(n_episodes=30)
        
        report = []
        report.append("=" * 60)
        report.append("VIDYA STRESS TEST ROBUSTNESS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Stress test results
        report.append("STRESS TEST RESULTS")
        report.append("-" * 40)
        
        for strategy, intensities in stress_results.items():
            report.append(f"\n{strategy.upper()} ATTACKS:")
            for intensity, metrics in intensities.items():
                report.append(
                    f"  Intensity {intensity}: "
                    f"Collapse Rate = {metrics['collapse_rate']:.2%}, "
                    f"Robustness = {metrics['robustness_score']:.2f}"
                )
                
        # Vulnerability analysis
        report.append("\n" + "-" * 40)
        report.append("VULNERABILITY ANALYSIS")
        report.append("-" * 40)
        
        sorted_vulns = sorted(
            vulnerabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for component, score in sorted_vulns:
            risk_level = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW"
            report.append(f"  {component}: {score:.2%} ({risk_level} RISK)")
            
        # Overall assessment
        report.append("\n" + "-" * 40)
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        avg_robustness = np.mean([
            m['robustness_score']
            for s in stress_results.values()
            for m in s.values()
        ])
        
        if avg_robustness > 0.8:
            assessment = "HIGHLY ROBUST"
        elif avg_robustness > 0.6:
            assessment = "MODERATELY ROBUST"
        elif avg_robustness > 0.4:
            assessment = "VULNERABLE"
        else:
            assessment = "HIGHLY VULNERABLE"
            
        report.append(f"Average Robustness Score: {avg_robustness:.2f}")
        report.append(f"Overall Assessment: {assessment}")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class RedTeamExercise:
    """
    Structured red team exercise for evaluating policy robustness.
    """
    
    def __init__(self, env_class, policy):
        self.env_class = env_class
        self.policy = policy
        
    def run_full_exercise(self, n_rounds: int = 5) -> Dict:
        """
        Run complete red team exercise with multiple attack rounds.
        """
        results = {
            'rounds': [],
            'policy_adaptations': [],
            'final_robustness': 0
        }
        
        for round_num in range(n_rounds):
            # Blue team (defender) tries to improve policy
            # Red team (adversary) finds new vulnerabilities
            
            # Find vulnerabilities with current policy
            runner = StressTestRunner(self.env_class(), self.policy)
            vulns = runner.find_vulnerabilities(n_episodes=20)
            
            # Red team focuses on most vulnerable component
            weakest = max(vulns, key=vulns.get)
            
            # Blue team adapts (simulated by increasing defense on weak point)
            adapted_policy = self._adapt_policy(weakest)
            
            # Test adapted policy
            runner = StressTestRunner(self.env_class(), adapted_policy)
            stress_results = runner.run_stress_test(n_episodes=20)
            
            round_result = {
                'round': round_num + 1,
                'vulnerability_found': weakest,
                'vulnerability_score': vulns[weakest],
                'adapted_policy': adapted_policy is not None,
                'robustness_after_adaptation': np.mean([
                    m['robustness_score']
                    for s in stress_results.values()
                    for m in s.values()
                ])
            }
            
            results['rounds'].append(round_result)
            
            # Update policy for next round
            if adapted_policy is not None:
                self.policy = adapted_policy
                results['policy_adaptations'].append(weakest)
                
        results['final_robustness'] = results['rounds'][-1]['robustness_after_adaptation']
        
        return results
        
    def _adapt_policy(self, vulnerability: str):
        """
        Simulate policy adaptation to address vulnerability.
        
        In practice, this would retrain or fine-tune the policy.
        For now, returns None to indicate conceptual adaptation.
        """
        # Placeholder: would implement actual policy adaptation
        return None
