"""
Meta-RL Implementation using MAML (Model-Agnostic Meta-Learning)

Enables the agent to learn how to quickly adapt to new scenarios with few gradient steps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import copy

from env.dropout_env import DropoutCommonsEnv
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from env.scenarios.pandemic_recovery import PandemicRecoveryScenario
from env.scenarios.conflict_zone import ConflictZoneScenario


class MetaPolicyNetwork(nn.Module):
    """
    Neural network policy with meta-learning capabilities.
    Can quickly adapt to new scenarios with few gradient steps.
    """
    
    def __init__(self, obs_dim: int = 13, act_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        # Main policy network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid()  # Actions in [0, 1]
        )
        
        # Context encoder for scenario embeddings
        self.context_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Value function for advantage estimation
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim + 32, hidden_dim),  # Include context
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning actions and value estimates.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            context: Optional scenario context [batch, 32]
            
        Returns:
            actions: [batch, act_dim]
            values: [batch, 1]
        """
        actions = self.policy_net(obs)
        
        if context is None:
            context = self.context_encoder(obs)
        
        combined = torch.cat([obs, context], dim=-1)
        values = self.value_net(combined)
        
        return actions, values
    
    def get_context(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract scenario context from observation."""
        return self.context_encoder(obs)


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning trainer for VIDYA.
    
    Outer loop: Learn meta-parameters that enable fast adaptation
    Inner loop: Adapt to specific scenario with few gradient steps
    """
    
    def __init__(
        self,
        policy: MetaPolicyNetwork,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        tasks_per_batch: int = 4,
        episodes_per_task: int = 10,
        device: str = 'cpu'
    ):
        self.policy = policy.to(device)
        self.meta_optimizer = optim.Adam(policy.parameters(), lr=meta_lr)
        
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.tasks_per_batch = tasks_per_batch
        self.episodes_per_task = episodes_per_task
        self.device = device
        
        self.meta_losses = []
        self.adaptation_performance = []
        
    def sample_task(self) -> Dict[str, Any]:
        """Sample a random scenario configuration (task)."""
        import numpy as np
        
        # Create scenarios with varying difficulty by modifying internal parameters
        scenario_types = [
            {'name': 'funding_crisis', 'class': FundingCutScenario, 'difficulty': np.random.choice(['easy', 'medium', 'hard'])},
            {'name': 'teacher_shortage', 'class': TeacherShortageScenario, 'difficulty': np.random.choice(['easy', 'medium', 'hard'])},
            {'name': 'pandemic_recovery', 'class': PandemicRecoveryScenario, 'difficulty': np.random.choice(['easy', 'medium', 'hard'])},
            {'name': 'conflict_zone', 'class': ConflictZoneScenario, 'difficulty': np.random.choice(['easy', 'medium', 'hard'])},
        ]
        
        task = scenario_types[np.random.randint(len(scenario_types))]
        task['scenario'] = task['class']()
        
        return task
    
    def collect_episodes(
        self,
        policy: MetaPolicyNetwork,
        task: Dict[str, Any],
        n_episodes: int
    ) -> List[Tuple]:
        """
        Collect episode rollouts from a task.
        
        Returns list of (obs, action, reward, next_obs, done) tuples.
        """
        episodes = []
        
        for _ in range(n_episodes):
            # Create environment with task-specific scenario
            env = DropoutCommonsEnv(
                scenario=task['scenario'],
                episode_length=100,
                noise_level=0.05
            )
            
            obs, info = env.reset()
            done = False
            episode_data = []
            
            while not done:
                # Get action from policy
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _ = policy(obs_tensor)
                    action = action.squeeze(0).cpu().numpy()
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_data.append((obs, action, reward, next_obs, done))
                obs = next_obs
            
            episodes.extend(episode_data)
        
        return episodes
    
    def compute_loss(
        self,
        policy: MetaPolicyNetwork,
        episodes: List[Tuple],
        gamma: float = 0.99
    ) -> torch.Tensor:
        """Compute policy gradient loss from episodes."""
        if not episodes:
            return torch.tensor(0.0, device=self.device)
        
        # Unpack episodes
        obs_list, action_list, reward_list, next_obs_list, done_list = zip(*episodes)
        
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(action_list)).to(self.device)
        reward_tensor = torch.FloatTensor(np.array(reward_list)).to(self.device)
        
        # Compute returns
        returns = []
        R = 0
        for r, d in zip(reversed(reward_list), reversed(done_list)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Compute policy loss
        actions_pred, values = policy(obs_tensor)
        
        # Policy gradient with value baseline
        advantages = returns_tensor.unsqueeze(-1) - values.detach()
        log_probs = -((actions_pred - action_tensor) ** 2)  # Simplified
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(-1), returns_tensor)
        
        return policy_loss + 0.5 * value_loss
    
    def inner_loop_adaptation(
        self,
        task: Dict[str, Any],
        n_steps: Optional[int] = None
    ) -> Tuple[MetaPolicyNetwork, float]:
        """
        Adapt policy to a specific task with few gradient steps (inner loop).
        
        Returns:
            adapted_policy: Policy adapted to task
            pre_adapt_loss: Loss before adaptation
        """
        n_steps = n_steps or self.inner_steps
        
        # Clone policy for this task
        adapted_policy = copy.deepcopy(self.policy)
        adapted_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)
        
        # Collect support set (adaptation data)
        support_episodes = self.collect_episodes(adapted_policy, task, n_episodes=3)
        
        # Compute pre-adaptation loss
        pre_adapt_loss = self.compute_loss(adapted_policy, support_episodes).item()
        
        # Inner loop: Adapt to task
        for step in range(n_steps):
            adapted_optimizer.zero_grad()
            loss = self.compute_loss(adapted_policy, support_episodes)
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_policy, pre_adapt_loss
    
    def meta_training_step(self) -> Dict[str, float]:
        """
        Single meta-training step (outer loop).
        
        1. Sample batch of tasks
        2. For each task: inner loop adaptation
        3. Evaluate on query set
        4. Meta-update to improve adaptation capability
        """
        self.meta_optimizer.zero_grad()
        
        meta_losses = []
        task_performances = []
        
        # Sample tasks
        tasks = [self.sample_task() for _ in range(self.tasks_per_batch)]
        
        for task in tasks:
            # Inner loop: Adapt to task
            adapted_policy, pre_loss = self.inner_loop_adaptation(task)
            
            # Collect query set (evaluation data)
            query_episodes = self.collect_episodes(adapted_policy, task, n_episodes=3)
            
            if query_episodes:
                # Compute post-adaptation loss (this is what we want to minimize)
                query_loss = self.compute_loss(adapted_policy, query_episodes)
                meta_losses.append(query_loss)
                task_performances.append(pre_loss)
        
        # Meta-update
        if meta_losses:
            meta_loss = torch.stack(meta_losses).mean()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            self.meta_losses.append(meta_loss.item())
            
            return {
                'meta_loss': meta_loss.item(),
                'avg_pre_adapt_loss': np.mean(task_performances),
                'tasks': len(tasks)
            }
        
        return {'meta_loss': 0.0, 'avg_pre_adapt_loss': 0.0, 'tasks': 0}
    
    def train(self, n_iterations: int = 1000, log_interval: int = 50) -> Dict[str, List]:
        """Run full meta-training."""
        print(f"Starting MAML training for {n_iterations} iterations...")
        print(f"Meta-lr: {self.meta_lr}, Inner-lr: {self.inner_lr}, Inner-steps: {self.inner_steps}")
        
        history = {
            'meta_loss': [],
            'pre_adapt_loss': [],
            'adaptation_gain': []
        }
        
        for iteration in range(n_iterations):
            metrics = self.meta_training_step()
            
            history['meta_loss'].append(metrics['meta_loss'])
            history['pre_adapt_loss'].append(metrics['avg_pre_adapt_loss'])
            
            if iteration % log_interval == 0:
                print(f"Iter {iteration}: meta_loss={metrics['meta_loss']:.4f}, "
                      f"pre_adapt_loss={metrics['avg_pre_adapt_loss']:.4f}")
        
        print("Meta-training complete!")
        return history
    
    def adapt_to_new_scenario(
        self,
        scenario_params: Dict[str, Any],
        n_gradient_steps: int = 10,
        n_episodes: int = 5
    ) -> MetaPolicyNetwork:
        """
        Adapt trained meta-policy to a completely new scenario.
        
        Args:
            scenario_params: Parameters of new scenario
            n_gradient_steps: Number of adaptation steps
            n_episodes: Episodes to collect for adaptation
            
        Returns:
            adapted_policy: Policy fine-tuned to new scenario
        """
        task = {
            'name': 'new_scenario',
            'params': scenario_params
        }
        
        adapted_policy, _ = self.inner_loop_adaptation(task, n_steps=n_gradient_steps)
        
        # Additional fine-tuning with more episodes
        episodes = self.collect_episodes(adapted_policy, task, n_episodes=n_episodes)
        
        optimizer = optim.Adam(adapted_policy.parameters(), lr=self.inner_lr)
        for _ in range(n_gradient_steps):
            optimizer.zero_grad()
            loss = self.compute_loss(adapted_policy, episodes)
            loss.backward()
            optimizer.step()
        
        return adapted_policy
    
    def save(self, path: str):
        """Save meta-trained policy."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'meta_optimizer_state': self.meta_optimizer.state_dict(),
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps
        }, path)
    
    def load(self, path: str):
        """Load meta-trained policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state'])


class ScenarioEmbedding:
    """
    Learn scenario embeddings to identify similar crisis situations.
    Useful for zero-shot transfer to unseen scenarios.
    """
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.scenarios = {}
        self.embeddings = {}
        
    def compute_embedding(self, scenario_trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Compute scenario embedding from trajectories.
        
        Args:
            scenario_trajectories: List of observation trajectories
            
        Returns:
            embedding: Scenario representation
        """
        # Simple statistical embedding
        all_obs = np.concatenate(scenario_trajectories, axis=0)
        
        embedding = np.concatenate([
            np.mean(all_obs, axis=0),
            np.std(all_obs, axis=0),
            np.max(all_obs, axis=0) - np.min(all_obs, axis=0)  # range
        ])[:self.embedding_dim]
        
        return embedding
    
    def find_similar_scenarios(
        self,
        new_scenario_embedding: np.ndarray,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar known scenarios.
        
        Returns:
            List of (scenario_name, similarity_score) tuples
        """
        similarities = []
        
        for name, embedding in self.embeddings.items():
            sim = np.dot(new_scenario_embedding, embedding) / (
                np.linalg.norm(new_scenario_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            similarities.append((name, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


def train_meta_rl(
    n_iterations: int = 500,
    save_path: str = "meta_policy.pt"
) -> MAMLTrainer:
    """
    Convenience function to train a meta-RL policy.
    
    Example:
        trainer = train_meta_rl(n_iterations=1000)
        adapted_policy = trainer.adapt_to_new_scenario({
            'initial_budget': 0.5,
            'funding_rate': 0.6
        })
    """
    policy = MetaPolicyNetwork(obs_dim=13, act_dim=8)
    
    trainer = MAMLTrainer(
        policy=policy,
        meta_lr=0.001,
        inner_lr=0.01,
        inner_steps=5,
        tasks_per_batch=4,
        episodes_per_task=10
    )
    
    history = trainer.train(n_iterations=n_iterations)
    trainer.save(save_path)
    
    print(f"Meta-policy saved to {save_path}")
    return trainer


if __name__ == "__main__":
    # Demo training
    trainer = train_meta_rl(n_iterations=100)
    
    # Test adaptation
    new_scenario = {
        'initial_budget': 0.3,
        'funding_rate': 0.4,
        'shock_probability': 0.4
    }
    
    adapted = trainer.adapt_to_new_scenario(new_scenario, n_gradient_steps=5)
    print("Adapted to new crisis scenario!")
