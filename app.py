"""
Hugging Face Spaces Demo for VIDYA
Interactive web interface to test trained policies and explore scenarios.
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import Optional, Dict, Any
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dropout_env import DropoutCommonsEnv
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from training.meta_rl import MetaPolicyNetwork, MAMLTrainer


class VIDYADemo:
    """Interactive demo for Hugging Face Spaces."""
    
    def __init__(self):
        self.current_model = None
        self.current_scenario = None
        self.simulation_history = []
        
    def load_model(self, model_type: str) -> str:
        """Load a pre-trained model."""
        try:
            if model_type == "meta_rl":
                # Load meta-trained policy
                policy = MetaPolicyNetwork()
                checkpoint_path = "checkpoints/meta_policy.pt"
                
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    policy.load_state_dict(checkpoint['policy_state_dict'])
                    self.current_model = policy
                    return "✅ Meta-RL policy loaded successfully!"
                else:
                    return "⚠️  Meta-policy not found. Using random policy."
            
            elif model_type == "ppo_standard":
                # Load standard PPO if available
                return "⚠️  PPO model not found. Please train a model first."
            
            else:
                self.current_model = None
                return "⚠️  Using random policy (no model loaded)"
                
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def create_scenario(
        self,
        scenario_type: str,
        difficulty: str,
        initial_budget: float,
        teacher_retention: float,
        enrollment_rate: float
    ) -> str:
        """Create a crisis scenario."""
        try:
            if scenario_type == "funding_crisis":
                self.current_scenario = {
                    'type': 'funding_crisis',
                    'params': {
                        'initial_budget': initial_budget / 100,
                        'teacher_retention': teacher_retention / 100,
                        'enrollment_rate': enrollment_rate / 100,
                        'difficulty': difficulty
                    }
                }
                
            elif scenario_type == "teacher_shortage":
                self.current_scenario = {
                    'type': 'teacher_shortage',
                    'params': {
                        'initial_budget': initial_budget / 100,
                        'teacher_retention': teacher_retention / 100,
                        'enrollment_rate': enrollment_rate / 100,
                        'difficulty': difficulty
                    }
                }
            
            return f"✅ Created {scenario_type.replace('_', ' ').title()} scenario (Difficulty: {difficulty})"
            
        except Exception as e:
            return f"❌ Error creating scenario: {str(e)}"
    
    def run_simulation(
        self,
        n_steps: int = 100,
        use_interventions: bool = True
    ) -> tuple:
        """
        Run simulation and return results.
        
        Returns:
            (status_message, trajectory_plot, metrics_plot, intervention_plot)
        """
        if self.current_scenario is None:
            return "❌ Please create a scenario first!", None, None, None
        
        try:
            # Create environment
            env = DropoutCommonsEnv(
                episode_length=n_steps,
                **self.current_scenario['params']
            )
            
            obs, info = env.reset()
            
            # Storage for visualization
            trajectories = {
                'enrollment': [],
                'dropout': [],
                'teacher_retention': [],
                'budget': [],
                'step': []
            }
            
            interventions = {
                'funding_boost': [],
                'teacher_incentive': [],
                'student_scholarship': [],
                'attendance_mandate': [],
                'resource_realloc': [],
                'transparency_report': [],
                'staff_hiring': [],
                'counseling_programs': [],
                'step': []
            }
            
            rewards = []
            done = False
            step = 0
            
            while not done and step < n_steps:
                # Get action from model or random
                if use_interventions and self.current_model is not None:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = self.current_model(obs_tensor)
                        action = action.squeeze(0).numpy()
                else:
                    # Random baseline
                    action = np.random.uniform(0, 0.3, size=8) if use_interventions else np.zeros(8)
                
                # Store intervention levels
                for i, key in enumerate(interventions.keys()):
                    if key != 'step':
                        interventions[key].append(action[i])
                
                interventions['step'].append(step)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store metrics
                state = env.state
                trajectories['enrollment'].append(state.enrollment_rate * 100)
                trajectories['dropout'].append(state.dropout_rate * 100)
                trajectories['teacher_retention'].append(state.teacher_retention * 100)
                trajectories['budget'].append(state.budget_utilization * 100)
                trajectories['step'].append(step)
                
                rewards.append(reward)
                obs = next_obs
                step += 1
            
            # Create plots
            trajectory_plot = self._create_trajectory_plot(trajectories)
            metrics_plot = self._create_metrics_plot(rewards, trajectories)
            intervention_plot = self._create_intervention_plot(interventions)
            
            # Summary
            final_enrollment = trajectories['enrollment'][-1]
            final_dropout = trajectories['dropout'][-1]
            final_teacher_ret = trajectories['teacher_retention'][-1]
            total_reward = sum(rewards)
            
            status = f"""
✅ Simulation Complete!

**Final Metrics:**
- Enrollment Rate: {final_enrollment:.1f}%
- Dropout Rate: {final_dropout:.1f}%
- Teacher Retention: {final_teacher_ret:.1f}%
- Total Reward: {total_reward:.2f}
- Episodes until collapse/termination: {step}

**Interpretation:**
{'✅ System maintained stability!' if final_enrollment > 60 else '⚠️  System experienced significant crisis'}
            """
            
            return status, trajectory_plot, metrics_plot, intervention_plot
            
        except Exception as e:
            return f"❌ Simulation error: {str(e)}", None, None, None
    
    def _create_trajectory_plot(self, trajectories: Dict) -> go.Figure:
        """Create state trajectory visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Enrollment Rate', 'Dropout Rate', 
                          'Teacher Retention', 'Budget Utilization'),
            vertical_spacing=0.15
        )
        
        colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B']
        metrics = ['enrollment', 'dropout', 'teacher_retention', 'budget']
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, color, (row, col) in zip(metrics, colors, positions):
            fig.add_trace(
                go.Scatter(
                    x=trajectories['step'],
                    y=trajectories[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=2)
                ),
                row=row, col=col
            )
            
            # Add threshold lines
            if metric == 'enrollment':
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=col)
            elif metric == 'dropout':
                fig.add_hline(y=25, line_dash="dash", line_color="red", row=row, col=col)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="System State Trajectories",
            template='plotly_white'
        )
        
        return fig
    
    def _create_metrics_plot(self, rewards: list, trajectories: Dict) -> go.Figure:
        """Create metrics summary plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cumulative Reward', 'System Stability')
        )
        
        # Cumulative reward
        cumsum_rewards = np.cumsum(rewards)
        fig.add_trace(
            go.Scatter(
                y=cumsum_rewards,
                mode='lines',
                name='Cumulative Reward',
                fill='tozeroy',
                line=dict(color='#8B5CF6')
            ),
            row=1, col=1
        )
        
        # Stability metric (enrollment - dropout)
        stability = np.array(trajectories['enrollment']) - np.array(trajectories['dropout'])
        fig.add_trace(
            go.Scatter(
                y=stability,
                mode='lines',
                name='Stability (Enroll - Dropout)',
                line=dict(color='#10B981')
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=300, template='plotly_white')
        
        return fig
    
    def _create_intervention_plot(self, interventions: Dict) -> go.Figure:
        """Create intervention usage heatmap."""
        # Extract intervention data
        intervention_names = [k for k in interventions.keys() if k != 'step']
        data = np.array([interventions[name] for name in intervention_names])
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=interventions['step'],
            y=[name.replace('_', ' ').title() for name in intervention_names],
            colorscale='Viridis',
            colorbar=dict(title='Intensity')
        ))
        
        fig.update_layout(
            title='Intervention Usage Over Time',
            xaxis_title='Time Step',
            yaxis_title='Intervention Type',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def compare_policies(self, n_steps: int = 50) -> tuple:
        """Compare RL policy vs random baseline."""
        if self.current_scenario is None:
            return "❌ Create a scenario first!", None
        
        try:
            # Run with RL policy
            rl_result = self._quick_simulation(use_model=True, n_steps=n_steps)
            
            # Run with random
            random_result = self._quick_simulation(use_model=False, n_steps=n_steps)
            
            # Create comparison plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=rl_result['enrollment'],
                mode='lines',
                name='RL Policy - Enrollment',
                line=dict(color='#3B82F6', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                y=random_result['enrollment'],
                mode='lines',
                name='Random - Enrollment',
                line=dict(color='#9CA3AF', width=2, dash='dash')
            ))
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Collapse Threshold")
            
            fig.update_layout(
                title='Policy Comparison: Enrollment Rate',
                xaxis_title='Time Step',
                yaxis_title='Enrollment Rate (%)',
                height=400,
                template='plotly_white'
            )
            
            comparison_text = f"""
**Policy Comparison ({n_steps} steps):**

**RL Policy:**
- Final Enrollment: {rl_result['enrollment'][-1]:.1f}%
- Final Dropout: {rl_result['dropout'][-1]:.1f}%
- Total Reward: {rl_result['total_reward']:.2f}

**Random Baseline:**
- Final Enrollment: {random_result['enrollment'][-1]:.1f}%
- Final Dropout: {random_result['dropout'][-1]:.1f}%
- Total Reward: {random_result['total_reward']:.2f}

**Improvement:**
- Enrollment: {rl_result['enrollment'][-1] - random_result['enrollment'][-1]:+.1f}%
- Reward: {rl_result['total_reward'] - random_result['total_reward']:.2f}
            """
            
            return comparison_text, fig
            
        except Exception as e:
            return f"❌ Comparison error: {str(e)}", None
    
    def _quick_simulation(self, use_model: bool, n_steps: int) -> Dict:
        """Quick simulation for comparison."""
        env = DropoutCommonsEnv(episode_length=n_steps, **self.current_scenario['params'])
        obs, _ = env.reset()
        
        enrollment = []
        dropout = []
        rewards = []
        
        for _ in range(n_steps):
            if use_model and self.current_model is not None:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.current_model(obs_tensor)
                    action = action.squeeze(0).numpy()
            else:
                action = np.random.uniform(0, 0.2, size=8)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            state = env.state
            enrollment.append(state.enrollment_rate * 100)
            dropout.append(state.dropout_rate * 100)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        return {
            'enrollment': enrollment,
            'dropout': dropout,
            'total_reward': sum(rewards)
        }


# Create Gradio interface
def create_spaces_demo() -> gr.Blocks:
    """Create the Hugging Face Spaces demo interface."""
    
    demo = VIDYADemo()
    
    with gr.Blocks(title="VIDYA - Educational Crisis Simulator", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎓 VIDYA: Educational System Crisis Simulator
        
        **Test AI policies for managing educational crises**
        
        This demo lets you:
        - Simulate crisis scenarios (funding cuts, teacher shortages, etc.)
        - Test RL-trained policies vs random interventions
        - Compare different crisis management strategies
        
        *Built with reinforcement learning and meta-learning (MAML)*
        """)
        
        with gr.Tab("🎮 Run Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Load Policy")
                    model_type = gr.Dropdown(
                        choices=["meta_rl", "ppo_standard", "random"],
                        value="meta_rl",
                        label="Select Model"
                    )
                    load_btn = gr.Button("Load Model", variant="primary")
                    load_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("### 2. Create Scenario")
                    scenario_type = gr.Dropdown(
                        choices=["funding_crisis", "teacher_shortage"],
                        value="funding_crisis",
                        label="Crisis Type"
                    )
                    difficulty = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="medium",
                        label="Difficulty"
                    )
                    
                    with gr.Row():
                        initial_budget = gr.Slider(30, 100, 70, label="Initial Budget (%)")
                        teacher_retention = gr.Slider(30, 100, 75, label="Teacher Retention (%)")
                    
                    enrollment_rate = gr.Slider(50, 100, 85, label="Initial Enrollment (%)")
                    
                    create_btn = gr.Button("Create Scenario", variant="secondary")
                    scenario_status = gr.Textbox(label="Scenario Status", interactive=False)
                    
                    gr.Markdown("### 3. Run")
                    n_steps = gr.Slider(50, 200, 100, step=10, label="Simulation Steps")
                    use_interventions = gr.Checkbox(True, label="Use AI Interventions")
                    run_btn = gr.Button("▶️ Run Simulation", variant="primary")
                
                with gr.Column(scale=2):
                    sim_status = gr.Textbox(label="Results", lines=8)
                    
                    with gr.Tabs():
                        with gr.Tab("📊 Trajectories"):
                            trajectory_plot = gr.Plot()
                        with gr.Tab("📈 Metrics"):
                            metrics_plot = gr.Plot()
                        with gr.Tab("🎛️ Interventions"):
                            intervention_plot = gr.Plot()
        
        with gr.Tab("⚖️ Compare Policies"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Compare RL vs Random Baseline")
                    compare_steps = gr.Slider(20, 100, 50, step=10, label="Steps")
                    compare_btn = gr.Button("Compare", variant="primary")
                
                with gr.Column(scale=2):
                    compare_status = gr.Textbox(label="Comparison Results", lines=12)
                    compare_plot = gr.Plot()
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About VIDYA
            
            VIDYA is a reinforcement learning environment for simulating educational system crises.
            
            ### Key Features:
            - **Meta-Learning (MAML)**: Policies that adapt quickly to new crisis scenarios
            - **LLM Integration**: Natural language interface for scenario creation
            - **RL-LLM Arbitration**: Combines RL actions with LLM reasoning
            
            ### Scenarios:
            1. **Funding Crisis**: Sudden budget cuts affecting operations
            2. **Teacher Shortage**: Mass departures creating staffing crisis
            3. **Pandemic Recovery**: Post-pandemic learning loss and enrollment drops
            4. **Conflict Zone**: Education under protracted armed conflict
            
            ### Interventions Available:
            - 💰 Funding boost
            - 👨‍🏫 Teacher incentives
            - 🎓 Student scholarships
            - 📊 Attendance mandates
            - 🔄 Resource reallocation
            - 📢 Transparency reports
            - 👥 Staff hiring
            - 💬 Counseling programs
            
            ---
            *Built with PyTorch, Stable-Baselines3, and Gradio*
            """)
        
        # Event handlers
        load_btn.click(
            fn=demo.load_model,
            inputs=[model_type],
            outputs=[load_status]
        )
        
        create_btn.click(
            fn=demo.create_scenario,
            inputs=[scenario_type, difficulty, initial_budget, teacher_retention, enrollment_rate],
            outputs=[scenario_status]
        )
        
        run_btn.click(
            fn=demo.run_simulation,
            inputs=[n_steps, use_interventions],
            outputs=[sim_status, trajectory_plot, metrics_plot, intervention_plot]
        )
        
        compare_btn.click(
            fn=demo.compare_policies,
            inputs=[compare_steps],
            outputs=[compare_status, compare_plot]
        )
    
    return app


if __name__ == "__main__":
    # For local testing
    app = create_spaces_demo()
    app.launch(server_name="0.0.0.0", server_port=7860)
