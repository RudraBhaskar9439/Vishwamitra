"""
DropoutCommonsEnv — Gymnasium environment for The Dropout Commons.

The meta-agent receives a system state and outputs intervention intensities.
Simulated agents (student, teacher, admin, policymaker) respond to those
interventions, updating the system state each step.
"""

import gymnasium as gym
import numpy as np
from .state import SystemState
from typing import Optional, Tuple, Dict, Any

from .spaces import OBS_DIM, ACT_DIM
from .spaces import make_observation_space, make_action_space
from agents.student_agent import StudentAgent
from agents.teacher_agent import TeacherAgent
from agents.admin_agent import AdminAgent
from agents.policymaker_agent import PolicymakerAgent
from env.scenarios.base_scenario import BaseScenario
from env.scenarios.funding_cut import FundingCutScenario

class DropoutCommonsEnv(gym.Env):
    """
    Multi-agent RL environment modeling education system dynamics.

    The gym.Env interface is for the META-AGENT (mechanism designer).
    Simulated agents run internally each step.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        scenario: Optional[BaseScenario] = None,
        episode_length: int = 100,
        noise_level: float = 0.05,
        render_modes: Optional[str] = None,
    ):
        super().__init__()
        self.scenario = scenario or FundingCutScenario()
        self.episode_length = episode_length
        self.noise_level = noise_level
        self.render_modes = render_modes

        self.observation_space = make_observation_space()
        self.action_space = make_action_space()

        # Simulated agents - Initialized in reset()
        self._student_agent: Optional[StudentAgent] = None
        self._teacher_agent: Optional[TeacherAgent] = None
        self._admin_agent: Optional[AdminAgent] = None
        self._policymaker_agent: Optional[PolicymakerAgent] = None

        self.state: Optional[SystemState] = None
        self._episode_log: list = []

#--------------------------------------------------------------------
# Gymnastic API
#--------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.state = self.scenario.initial_state(rng=self.np_random)

        self._student_agent = StudentAgent(state=self.state, rng=self.np_random)
        self._teacher_agent = TeacherAgent(state=self.state, rng=self.np_random)
        self._admin_agent = AdminAgent(state=self.state, rng=self.np_random)
        self._policymaker_agent = PolicymakerAgent(state=self.state, rng=self.np_random)

        self._episode_log = []
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.state is not None, "Call reset() before step()."
        action = np.clip(action, 0.0, 1.0)

        # 1. Apply meta-agent interventions -> modify incentive signals
        incentives = self._parse_action(action)

        # 2. Run each simulated agent for one timestep
        self._student_agent.step(self.state, incentives)
        self._teacher_agent.step(self.state, incentives)
        self._admin_agent.step(self.state, incentives)
        self._policymaker_agent.step(self.state, incentives)

        # 3. Apply scenario-level shocks (funding cuts, etc.)
        self.scenario.apply_shock(self.state, self.state.step)

        # 4. Add observation noise
        self._apply_noise()

        # 5. Tick step counter
        self.state.step += 1
        self.state.health_history.append(self.state.health_score)

        # 6. Compute reward
        reward = self._compute_reward(incentives)

        # 7. Termination
        terminated = self._check_collapse()
        truncated = self.state.step >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()
        info["health_score"] = self.state.health_score
        info["incentive_cost"] = incentives["total_cost"]

        self._episode_log.append({
            "step": self.state.step,
            "health_score": self.state.health_score,
            "dropout_rate": self.state.dropout_rate,
            "teacher_retention": self.state.teacher_retention,
            **incentives,
        })

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            s = self.state
            print(
                f"Step {s.step:3d} | Health={s.health_score:.3f} | "
                f"Dropout={s.dropout_rate:.2f} | TeacherRetention={s.teacher_retention:.2f} | "
                f"Budget={s.budget_remaining:,.0f}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_action(self, action: np.ndarray) -> Dict[str, float]:
        """Map raw action vector to named incentive dict with costs."""
        COSTS = [50000, 80000, 30000, 10000, 40000, 5000, 120000, 25000]
        labels = [
            "funding_boost", "teacher_incentive", "student_scholarship",
            "attendance_mandate", "resource_realloc", "transparency_report",
            "staff_hiring", "counseling_programs",
        ]
        incentives = {labels[i]: float(action[i]) for i in range(ACT_DIM)}
        total_cost = sum(action[i] * COSTS[i] for i in range(ACT_DIM))
        incentives["total_cost"] = total_cost
        return incentives

    def _apply_noise(self):
        """Simulate data corruption and reporting noise."""
        s = self.state
        noise = self.np_random.normal(0, self.noise_level, size=5)
        s.enrollment_rate = float(np.clip(s.enrollment_rate + noise[0] * 0.02, 0, 1))
        s.attendance_rate = float(np.clip(s.attendance_rate + noise[1] * 0.02, 0, 1))
        s.dropout_rate = float(np.clip(s.dropout_rate + noise[2] * 0.01, 0, 0.5))
        s.teacher_burnout = float(np.clip(s.teacher_burnout + noise[3] * 0.01, 0, 1))
        s.student_engagement = float(np.clip(s.student_engagement + noise[4] * 0.02, 0, 1))

    def _compute_reward(self, incentives: Dict) -> float:
        return _reward_fn(self.state, incentives)

    def _check_collapse(self) -> bool:
        """Episode terminates early if system collapses."""
        s = self.state
        return (
            s.dropout_rate > 0.50
            or s.teacher_retention < 0.20
            or s.budget_remaining < -500_000
            or s.enrollment_rate < 0.30
        )

    def _get_obs(self) -> np.ndarray:
        return self.state.to_obs_array()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.state.step,
            "dropout_rate": self.state.dropout_rate,
            "teacher_retention": self.state.teacher_retention,
            "budget_remaining": self.state.budget_remaining,
        }

    def get_episode_log(self) -> list:
        return self._episode_log


def _reward_fn(state, incentives: dict) -> float:
    """Compute reward based on state health and incentive costs."""
    cost = incentives.get("total_cost", 0.0)
    return float(np.clip(
        -2.0 * state.dropout_rate
        + 1.0 * (state.teacher_retention - 0.7)
        + state.student_engagement * 0.5
        - 0.001 * cost / 50000,
        -2.0, 2.0
    ))