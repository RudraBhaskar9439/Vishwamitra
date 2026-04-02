"""
Main training script for the Dropout Commons meta-agent.

Usage:
    python -m training.train
    python -m training.train --total_timesteps 2000000
"""

import argparse
import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from env.dropout_env import DropoutCommonsEnv
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from env.scenarios.indian_context import IndianContextScenario
from training.config import TrainingConfig
from training.callbacks import HealthScoreCallback
from training.curriculum import CurriculumScheduler


SCENARIOS = {
    "funding_cut": FundingCutScenario,
    "teacher_shortage": TeacherShortageScenario,
    "indian_context": IndianContextScenario,
}

def make_env(config: TrainingConfig, rank: int = 0):
    def _init():
        scenario_cls = SCENARIOS.get(config.env.scenario, FundingCutScenario)
        env = DropoutCommonsEnv(
            scenario=scenario_cls(),
            episode_length=config.env.episode_length,
            noise_level=config.env.noise_level,
        )
        env = Monitor(env, filename=f"logs/env_{rank}.csv")
        return env
    return _init


def train(config: TrainingConfig):
    run_id = f"{config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --- Vectorized parallel environments ---
    vec_env = make_vec_env(
        make_env(config),
        n_envs=config.n_envs,
        seed=config.seed,
    )
    # Normalize observations and rewards for stable training
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Evaluation environment (single, unnormalized) ---
    eval_env = make_vec_env(make_env(config), n_envs=1, seed=config.seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # --- PPO meta-agent ---
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        verbose=1,
        tensorboard_log=f"logs/tensorboard/{run_id}",
        seed=config.seed,
        policy_kwargs=dict(net_arch=[256, 256, 128]),  # 3-layer MLP
    )

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=config.save_interval // config.n_envs,
        save_path=config.checkpoint_dir,
        name_prefix=run_id,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=config.checkpoint_dir,
        log_path="logs/eval",
        eval_freq=10_000 // config.n_envs,
        n_eval_episodes=20,
        deterministic=True,
    )
    health_cb = HealthScoreCallback()

    # --- Train ---
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_cb, eval_cb, health_cb],
        log_interval=config.log_interval,
        reset_num_timesteps=True,
        tb_log_name=run_id,
    )

    # Save final model + normalization stats
    model.save(os.path.join(config.checkpoint_dir, f"{run_id}_final"))
    vec_env.save(os.path.join(config.checkpoint_dir, f"{run_id}_vecnorm.pkl"))
    print(f"Training complete. Model saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--run_name", type=str, default="dropout_commons_v1")
    parser.add_argument("--scenario", type=str, default="funding_cut",
                        choices=["funding_cut", "teacher_shortage", "indian_context"])
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = TrainingConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        run_name=args.run_name,
    )
    cfg.env.scenario = args.scenario
    cfg.ppo.learning_rate = args.learning_rate
    train(cfg)