from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class EnvConfig:
    episode_length: int = 100
    noise_level: float = 0.05
    # All VIDYA scenarios:
    # funding_cut, teacher_shortage, indian_context, rural_india
    # pandemic_recovery, learning_loss, hybrid_learning
    # conflict_zone, natural_disaster, displacement_crisis
    scenario: str = "funding_cut"
    use_adversarial_agent: bool = False
    adversarial_intensity: float = 0.5

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048             # steps before each policy update
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995            # high gamma for long-horizon rewards
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01          # encourage exploration of intervention space
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

@dataclass
class CurriculumConfig:
    enabled: bool = False
    scenario_sequence: List[str] = field(default_factory=lambda: [
        "funding_cut",
        "teacher_shortage",
        "pandemic_recovery",
        "conflict_zone"
    ])
    timesteps_per_scenario: int = 250_000
    success_threshold: float = 0.7  # Collapse rate below this to advance

@dataclass
class CollapseDetectionConfig:
    enabled: bool = True
    method: str = "ensemble"  # critical_slowing_down, correlation_network, shock_propagation, ensemble
    warning_steps: int = 10
    training_data_size: int = 1000

@dataclass
class GameTheoryConfig:
    enabled: bool = False
    nash_equilibrium_interval: int = 1000  # Steps between NE computation
    mechanism_design_interval: int = 5000

@dataclass
class HuggingFaceConfig:
    enabled: bool = False
    repo_id: Optional[str] = None
    push_to_hub: bool = False
    save_best: bool = True
    best_metric: str = "collapse_rate"

@dataclass
class TrainingConfig:
    """VIDYA Training Configuration"""
    # Basic training
    total_timesteps: int = 1_000_000
    n_envs: int = 8                 # parallel envs for faster rollouts
    log_interval: int = 10
    save_interval: int = 50_000
    checkpoint_dir: str = "checkpoints/"
    run_name: str = "vidya_v1"
    seed: int = 42
    
    # Sub-configs
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    collapse_detection: CollapseDetectionConfig = field(default_factory=CollapseDetectionConfig)
    game_theory: GameTheoryConfig = field(default_factory=GameTheoryConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    # Metadata
    project_name: str = "VIDYA"
    description: str = "Multi-Agent RL for Educational System Collapse Prevention"
    tags: List[str] = field(default_factory=lambda: ["vidya", "education", "rl", "multi-agent"])
    
    def get_scenario_class(self):
        """Get the scenario class based on config."""
        from env.scenarios.funding_cut import FundingCutScenario
        from env.scenarios.teacher_shortage import TeacherShortageScenario
        from env.scenarios.indian_context import IndianContextScenario
        from env.scenarios.pandemic_recovery import PandemicRecoveryScenario, LearningLossScenario, HybridLearningScenario
        from env.scenarios.conflict_zone import ConflictZoneScenario, NaturalDisasterScenario, DisplacementCrisisScenario
        
        scenario_map = {
            'funding_cut': FundingCutScenario,
            'teacher_shortage': TeacherShortageScenario,
            'indian_context': IndianContextScenario,
            'rural_india': IndianContextScenario,
            'pandemic_recovery': PandemicRecoveryScenario,
            'learning_loss': LearningLossScenario,
            'hybrid_learning': HybridLearningScenario,
            'conflict_zone': ConflictZoneScenario,
            'natural_disaster': NaturalDisasterScenario,
            'displacement_crisis': DisplacementCrisisScenario,
        }
        
        return scenario_map.get(self.env.scenario, FundingCutScenario)