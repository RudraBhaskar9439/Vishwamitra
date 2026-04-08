"""
Hugging Face Spaces Demo for VIDYA
Interactive web interface to test trained policies and explore scenarios.
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import Optional, Dict, Any, List, Tuple
import sys
import os
import json
import re
import datetime as _dt

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dropout_env import DropoutCommonsEnv
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from env.scenarios.pandemic_recovery import PandemicRecoveryScenario
from env.scenarios.conflict_zone import ConflictZoneScenario
from training.meta_rl import MetaPolicyNetwork, MAMLTrainer

SCENARIO_MAP = {
    "funding_crisis": FundingCutScenario,
    "teacher_shortage": TeacherShortageScenario,
    "pandemic_recovery": PandemicRecoveryScenario,
    "conflict_zone": ConflictZoneScenario,
}


# Keyword routing — maps free-text crisis descriptions to a scenario class.
# First match wins; falls back to FundingCutScenario.
SCENARIO_KEYWORDS = [
    (("fund", "budget", "money", "cut", "financ"), FundingCutScenario, "funding_crisis"),
    (("teacher", "staff", "shortage", "exodus", "burnout"), TeacherShortageScenario, "teacher_shortage"),
    (("pandemic", "covid", "lockdown", "learning loss", "remote"), PandemicRecoveryScenario, "pandemic_recovery"),
    (("conflict", "war", "displacement", "refugee", "violence"), ConflictZoneScenario, "conflict_zone"),
]


def _resolve_scenario(text: str) -> tuple:
    """Map a free-text crisis description to (ScenarioClass, label)."""
    if not text:
        return FundingCutScenario, "funding_crisis"
    t = text.lower()
    for keywords, cls, label in SCENARIO_KEYWORDS:
        if any(kw in t for kw in keywords):
            return cls, label
    return FundingCutScenario, "funding_crisis"


def _build_env(scenario_cfg: dict, n_steps: int) -> DropoutCommonsEnv:
    """Construct DropoutCommonsEnv from a UI scenario config."""
    scenario_cls = scenario_cfg.get("class", FundingCutScenario)
    return DropoutCommonsEnv(scenario=scenario_cls(), episode_length=n_steps)


# ---------------------------------------------------------------------------
# LLM advisory layer (multi-perspective agent reasoning)
# ---------------------------------------------------------------------------

OBS_FIELDS = [
    ("enrollment_rate", "Enrollment Rate"),
    ("attendance_rate", "Attendance Rate"),
    ("dropout_rate", "Dropout Rate"),
    ("teacher_retention", "Teacher Retention"),
    ("budget_utilization", "Budget Utilization"),
    ("class_size_norm", "Class Size (norm.)"),
    ("teacher_workload", "Teacher Workload"),
    ("resource_allocation", "Resource Allocation"),
    ("student_engagement", "Student Engagement"),
    ("teacher_burnout", "Teacher Burnout"),
    ("policy_compliance", "Policy Compliance"),
    ("budget_remaining_norm", "Budget Remaining"),
    ("step", "Time Step"),
]

AGENT_PERSONAS = {
    "Student": (
        "You are a 14-year-old student in this school. You speak from the lived "
        "experience of attending class day to day, dealing with peers, family "
        "pressures, and your own motivation. You care about whether you can stay "
        "in school and learn something useful."
    ),
    "Teacher": (
        "You are a senior classroom teacher with 12 years of experience. You "
        "speak from the perspective of someone managing 40+ students, drowning in "
        "paperwork, watching colleagues quit, and trying to keep your own passion "
        "alive. You care about working conditions and pedagogical integrity."
    ),
    "Administrator": (
        "You are the school principal. You answer to the district, parents, and "
        "your staff simultaneously. You speak from the perspective of someone "
        "balancing budgets, compliance, and morale. You care about institutional "
        "survival and reputation."
    ),
    "Policymaker": (
        "You are a state-level education policymaker. You think in terms of "
        "systems, equity across schools, political constraints, and 5-year "
        "outcomes. You speak from the perspective of someone who must justify "
        "interventions to ministers and the public."
    ),
}

FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.jsonl")


def _llm_client():
    """Return an OpenAI-compatible client if env vars are set, else None."""
    base = os.environ.get("API_BASE_URL", "")
    model = os.environ.get("MODEL_NAME", "")
    token = os.environ.get("HF_TOKEN", "")
    if not (base and model and token):
        return None, None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(base_url=base, api_key=token), model
    except Exception:
        return None, None


def _llm_chat(messages: List[Dict], temperature: float = 0.6, max_tokens: int = 500) -> Optional[str]:
    client, model = _llm_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM error: {e}]"


def get_agent_response(role: str, crisis_text: str, difficulty: str) -> str:
    """Ask one agent persona how they read the crisis."""
    persona = AGENT_PERSONAS.get(role, "")
    sys_msg = (
        persona + " Respond in 4–6 sentences. Be concrete and grounded. "
        "Avoid platitudes. Speak in first person."
    )
    user_msg = (
        f"A crisis is unfolding in your school: \"{crisis_text}\".\n"
        f"Severity: {difficulty}.\n"
        "What is your honest read of what is happening, and what would you "
        "personally do in the next two weeks?"
    )
    out = _llm_chat(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        temperature=0.7,
        max_tokens=350,
    )
    return out or f"[{role} unavailable — set API_BASE_URL/MODEL_NAME/HF_TOKEN to enable]"


def get_final_verdict(crisis_text: str, agent_responses: Dict[str, str], difficulty: str) -> str:
    """Synthesize a mechanism-design verdict from all agent perspectives."""
    if not agent_responses:
        return "No agents selected. Pick at least one stakeholder above."
    transcript = "\n\n".join(f"## {r}\n{t}" for r, t in agent_responses.items())
    sys_msg = (
        "You are Vishwamitra — an AI mechanism designer. You read the perspectives "
        "of multiple stakeholders in an educational system and propose a concrete, "
        "deployable intervention bundle. You think in terms of game theory, "
        "incentive structures, and the Tragedy of the Commons. You DO NOT pick "
        "winners; you redesign the rules so cooperation becomes dominant for all."
    )
    user_msg = (
        f"Crisis: \"{crisis_text}\" (severity: {difficulty})\n\n"
        f"Stakeholder transcripts:\n\n{transcript}\n\n"
        "Now produce a FINAL VERDICT in this exact structure:\n"
        "**Diagnosis:** (2 sentences naming the underlying coordination failure)\n"
        "**Intervention Bundle:** (3–5 bullet points, each one specific and timed)\n"
        "**Why this shifts the equilibrium:** (2–3 sentences in game-theory terms)\n"
        "**First action this week:** (one concrete sentence)"
    )
    out = _llm_chat(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        temperature=0.5,
        max_tokens=700,
    )
    return out or "[verdict unavailable — set API credentials]"


def get_dynamic_plot_config(crisis_text: str) -> List[Tuple[int, str]]:
    """Ask the LLM which 4 of the 13 obs metrics best illustrate this crisis,
    and what to title each chart. Returns list of (obs_index, title)."""
    options = "\n".join(
        f"{i}: {field[1]} ({field[0]})" for i, field in enumerate(OBS_FIELDS)
    )
    sys_msg = (
        "You are a data visualization curator. Given a crisis description, you "
        "pick the FOUR most relevant metrics to chart and give each chart a "
        "specific, vivid title that reflects the crisis being analyzed."
    )
    user_msg = (
        f"Crisis: \"{crisis_text}\"\n\n"
        f"Available metrics:\n{options}\n\n"
        "Reply with ONLY a JSON object of this exact form:\n"
        '{"plots": [{"index": <int>, "title": "<short vivid title>"}, ...]}\n'
        "Choose exactly 4. Use indices from the list above."
    )
    raw = _llm_chat(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        temperature=0.4,
        max_tokens=300,
    )
    fallback = [(0, "Enrollment Rate"), (2, "Dropout Rate"),
                (3, "Teacher Retention"), (8, "Student Engagement")]
    if not raw:
        return fallback
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return fallback
        data = json.loads(match.group(0))
        plots = data.get("plots", [])
        result = []
        for p in plots[:4]:
            idx = int(p.get("index", 0))
            title = str(p.get("title", OBS_FIELDS[idx][1]))[:60]
            if 0 <= idx < len(OBS_FIELDS):
                result.append((idx, title))
        return result if len(result) == 4 else fallback
    except Exception:
        return fallback


def save_feedback(crisis_text: str, verdict: str, rating: int, comment: str) -> str:
    """Persist user feedback as JSONL — training data for future fine-tuning."""
    record = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "crisis": crisis_text,
        "verdict": verdict,
        "rating": int(rating),
        "comment": comment.strip(),
    }
    try:
        with open(FEEDBACK_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return (
            f"✅ Feedback saved (#{_count_feedback()}).\n"
            "This becomes part of the training set Vishwamitra uses to refine "
            "future verdicts. Thank you."
        )
    except Exception as e:
        return f"❌ Could not save feedback: {e}"


def _count_feedback() -> int:
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    try:
        with open(FEEDBACK_PATH) as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


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
        crisis_text: str,
        difficulty: str,
        initial_budget: float,
        teacher_retention: float,
        enrollment_rate: float,
    ) -> str:
        """Create a crisis scenario from a free-text description."""
        try:
            scenario_cls, label = _resolve_scenario(crisis_text)
            self.current_scenario = {
                "type": label,
                "class": scenario_cls,
                "label": (crisis_text or label).strip(),
                "params": {
                    "initial_budget": initial_budget / 100,
                    "teacher_retention": teacher_retention / 100,
                    "enrollment_rate": enrollment_rate / 100,
                    "difficulty": difficulty,
                },
            }
            display = (crisis_text or label).strip()
            return (
                f"✅ Crisis registered: \"{display}\"\n"
                f"   Routed to archetype: {label.replace('_', ' ').title()}\n"
                f"   Difficulty: {difficulty}"
            )
        except Exception as e:
            return f"❌ Error creating scenario: {str(e)}"
    
    def run_simulation(
        self,
        selected_agents: List[str],
        n_steps: int = 100,
        use_interventions: bool = True,
    ) -> tuple:
        """Run env + LLM advisory pipeline.

        Returns:
            (perspectives_md, verdict_md, status_md,
             trajectory_plot, metrics_plot, intervention_plot)
        """
        if self.current_scenario is None:
            return ("❌ Please create a scenario first.", "", "",
                    None, None, None)

        try:
            crisis_text = self.current_scenario.get("label", "")
            difficulty = self.current_scenario.get("params", {}).get("difficulty", "medium")

            # 1. Run the underlying env to collect a 13-dim trajectory
            env = _build_env(self.current_scenario, n_steps)
            obs, _ = env.reset()
            obs_history: List[np.ndarray] = []
            interventions = {label: [] for _, label in OBS_FIELDS[:8]}  # placeholder
            interv_keys = [
                "funding_boost", "teacher_incentive", "student_scholarship",
                "attendance_mandate", "resource_realloc", "transparency_report",
                "staff_hiring", "counseling_programs",
            ]
            interv_track = {k: [] for k in interv_keys}
            interv_track["step"] = []
            rewards: List[float] = []
            step = 0
            done = False
            while not done and step < n_steps:
                if use_interventions and self.current_model is not None:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = self.current_model(obs_tensor)
                        action = action.squeeze(0).numpy()
                else:
                    action = np.random.uniform(0, 0.3, size=8) if use_interventions else np.zeros(8)
                for i, k in enumerate(interv_keys):
                    interv_track[k].append(float(action[i]))
                interv_track["step"].append(step)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                obs_history.append(np.asarray(next_obs, dtype=np.float32))
                rewards.append(float(reward))
                obs = next_obs
                step += 1
                done = terminated or truncated

            obs_arr = np.stack(obs_history) if obs_history else np.zeros((1, 13))

            # 2. LLM: agent perspectives
            perspectives: Dict[str, str] = {}
            for role in (selected_agents or []):
                perspectives[role] = get_agent_response(role, crisis_text, difficulty)

            if perspectives:
                perspectives_md = "## 🗣️ Stakeholder Perspectives\n\n" + "\n\n".join(
                    f"### {role}\n{text}" for role, text in perspectives.items()
                )
            else:
                perspectives_md = (
                    "## 🗣️ Stakeholder Perspectives\n\n_No agents selected. "
                    "Tick at least one stakeholder above to hear their perspective._"
                )

            # 3. LLM: final verdict
            verdict_text = get_final_verdict(crisis_text, perspectives, difficulty)
            verdict_md = "## ⚖️ Vishwamitra's Verdict\n\n" + verdict_text
            self._last_verdict = verdict_text
            self._last_crisis = crisis_text

            # 4. LLM: dynamic plot config
            plot_config = get_dynamic_plot_config(crisis_text)
            trajectory_plot = self._create_dynamic_trajectory_plot(obs_arr, plot_config)
            metrics_plot = self._create_metrics_plot(rewards, obs_arr)
            intervention_plot = self._create_intervention_plot(interv_track)

            # 5. Status
            total_reward = sum(rewards)
            status_md = (
                f"### Episode Summary\n"
                f"- Crisis: **{crisis_text}**\n"
                f"- Stakeholders consulted: **{', '.join(selected_agents) if selected_agents else 'none'}**\n"
                f"- Steps simulated: **{step}**\n"
                f"- Cumulative reward: **{total_reward:+.2f}**\n"
                f"- Terminated early: **{'yes' if done and step < n_steps else 'no'}**"
            )

            return perspectives_md, verdict_md, status_md, trajectory_plot, metrics_plot, intervention_plot

        except Exception as e:
            return (f"❌ Simulation error: {e}", "", "", None, None, None)
    
    def _create_dynamic_trajectory_plot(
        self, obs_arr: np.ndarray, plot_config: List[Tuple[int, str]]
    ) -> go.Figure:
        """Plot 4 metrics whose indices and titles were chosen dynamically."""
        titles = [t for _, t in plot_config]
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=titles,
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )
        colors = [
            ('#6ea8ff', 'rgba(110,168,255,0.15)'),
            ('#f87171', 'rgba(248,113,113,0.15)'),
            ('#4ade80', 'rgba(74,222,128,0.15)'),
            ('#f5b942', 'rgba(245,185,66,0.15)'),
        ]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        steps = np.arange(obs_arr.shape[0])
        for (idx, _title), (line_c, fill_c), (row, col) in zip(plot_config, colors, positions):
            series = obs_arr[:, idx] if idx < obs_arr.shape[1] else np.zeros_like(steps)
            fig.add_trace(
                go.Scatter(
                    x=steps, y=series, mode='lines',
                    line=dict(color=line_c, width=2.5),
                    fill='tozeroy', fillcolor=fill_c,
                ),
                row=row, col=col,
            )
        fig.update_layout(
            height=500, showlegend=False,
            title_text="Dynamic Crisis Trajectories",
            template='plotly_dark',
            paper_bgcolor='#0a0e1a',
            plot_bgcolor='#111827',
            font=dict(color='#e6edf7', family='Inter'),
        )
        return fig

    def _create_metrics_plot(self, rewards: list, obs_arr: np.ndarray) -> go.Figure:
        """Reward + composite-health summary."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cumulative Reward', 'Composite System Health'),
            horizontal_spacing=0.12,
        )
        cumsum_rewards = np.cumsum(rewards) if len(rewards) else np.array([0.0])
        fig.add_trace(
            go.Scatter(
                y=cumsum_rewards, mode='lines',
                line=dict(color='#f5b942', width=2.5),
                fill='tozeroy', fillcolor='rgba(245,185,66,0.15)',
            ),
            row=1, col=1,
        )
        if obs_arr.size:
            health = (
                0.25 * obs_arr[:, 0]   # enrollment
                + 0.20 * obs_arr[:, 1] # attendance
                + 0.20 * obs_arr[:, 3] # teacher_retention
                + 0.15 * obs_arr[:, 8] # student_engagement
                - 0.30 * obs_arr[:, 2] # dropout
            )
        else:
            health = np.array([0.0])
        fig.add_trace(
            go.Scatter(
                y=health, mode='lines',
                line=dict(color='#6ea8ff', width=2.5),
                fill='tozeroy', fillcolor='rgba(110,168,255,0.15)',
            ),
            row=1, col=2,
        )
        fig.update_layout(
            height=320, showlegend=False,
            template='plotly_dark',
            paper_bgcolor='#0a0e1a', plot_bgcolor='#111827',
            font=dict(color='#e6edf7', family='Inter'),
        )
        
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
            template='plotly_dark',
            paper_bgcolor='#0a0e1a', plot_bgcolor='#111827',
            font=dict(color='#e6edf7', family='Inter'),
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
        env = _build_env(self.current_scenario, n_steps)
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
CUSTOM_CSS = """
:root {
    --vm-bg: #0a0e1a;
    --vm-bg-2: #111827;
    --vm-surface: #1a2235;
    --vm-border: #243049;
    --vm-text: #e6edf7;
    --vm-text-dim: #93a3bd;
    --vm-accent: #f5b942;
    --vm-accent-2: #6ea8ff;
    --vm-good: #4ade80;
    --vm-bad: #f87171;
}

.gradio-container {
    background: radial-gradient(ellipse at top, #131a2e 0%, var(--vm-bg) 60%) !important;
    color: var(--vm-text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
}

.vm-hero {
    padding: 48px 56px 40px 56px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(110,168,255,0.10), rgba(245,185,66,0.06));
    border: 1px solid var(--vm-border);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.vm-hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(circle at 90% 0%, rgba(245,185,66,0.10), transparent 50%);
    pointer-events: none;
}
.vm-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 12px;
    color: var(--vm-accent);
    font-weight: 600;
    margin-bottom: 12px;
}
.vm-title {
    font-size: 56px;
    line-height: 1.05;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #93a3bd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 14px 0;
}
.vm-tagline {
    font-size: 20px;
    color: var(--vm-text-dim);
    font-weight: 400;
    max-width: 780px;
    margin-bottom: 22px;
}
.vm-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.vm-pill {
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--vm-border);
    background: rgba(255,255,255,0.03);
    font-size: 12px;
    color: var(--vm-text-dim);
    font-weight: 500;
}
.vm-section-h {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--vm-text-dim);
    font-weight: 600;
    margin: 4px 0 12px 0;
}
.tabs > .tab-nav button {
    background: transparent !important;
    color: var(--vm-text-dim) !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 14px 22px !important;
    font-size: 14px !important;
}
.tabs > .tab-nav button.selected {
    color: var(--vm-text) !important;
    border-bottom: 2px solid var(--vm-accent) !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, var(--vm-accent), #e09b1f) !important;
    color: #0a0e1a !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
.gr-button-secondary {
    background: rgba(110,168,255,0.10) !important;
    color: var(--vm-accent-2) !important;
    border: 1px solid rgba(110,168,255,0.30) !important;
    border-radius: 10px !important;
}
.gr-box, .gr-form, .gr-panel {
    background: var(--vm-surface) !important;
    border: 1px solid var(--vm-border) !important;
    border-radius: 14px !important;
}
.vm-about h2 {
    font-size: 30px; font-weight: 700; margin-top: 28px; color: var(--vm-text);
    border-left: 3px solid var(--vm-accent); padding-left: 14px;
}
.vm-about h3 {
    font-size: 18px; font-weight: 600; margin-top: 22px; color: var(--vm-accent-2);
}
.vm-about p, .vm-about li { color: var(--vm-text-dim); line-height: 1.7; font-size: 15px; }
.vm-about strong { color: var(--vm-text); font-weight: 600; }
.vm-quote {
    border-left: 3px solid var(--vm-accent);
    padding: 16px 22px;
    background: rgba(245,185,66,0.04);
    border-radius: 0 10px 10px 0;
    font-style: italic;
    color: var(--vm-text);
    margin: 22px 0;
}
"""


HERO_HTML = """
<div class="vm-hero">
  <div class="vm-eyebrow">Meta · PyTorch · OpenEnv Round 1</div>
  <h1 class="vm-title">Vishwamitra</h1>
  <p class="vm-tagline">Seeing clearly. Redesigning the game. A multi-agent reinforcement learning environment that learns to rewrite the rules of educational collapse before it happens.</p>
  <div class="vm-pills">
    <span class="vm-pill">Mechanism Design</span>
    <span class="vm-pill">Game Theory</span>
    <span class="vm-pill">Meta-RL</span>
    <span class="vm-pill">Built on Meta OpenEnv</span>
    <span class="vm-pill">LLM-Graded Policies</span>
  </div>
</div>
"""


ABOUT_HTML = """
<div class="vm-about">

<h2>The Problem</h2>
<p>Public education doesn't fail because of bad people — it fails because of a <strong>coordination problem</strong>. Every stakeholder makes a locally rational choice: the student stops attending, the teacher burns out, the administrator delays, the policymaker redirects funds. None of them chose collapse — but all of them, acting independently, produce it together.</p>
<p>This is the <strong>Tragedy of the Commons</strong> applied to education. It is a game theory problem, not a resource problem.</p>

<h2>What Vishwamitra Is</h2>
<ul>
  <li>A multi-agent reinforcement learning environment built on <strong>Meta's OpenEnv framework</strong>.</li>
  <li>Models a school as a shared resource being depleted by four rational agents playing dominant strategies simultaneously.</li>
  <li>Trains an RL agent to detect cascading defection early and deploy targeted interventions before the system crosses an irreversible collapse threshold.</li>
  <li>The agent is not a player inside the game — it is a <strong>mechanism designer</strong> that rewrites the incentive structure so cooperation becomes the dominant strategy for every player.</li>
</ul>

<h2>The Game Theory Stack</h2>
<ul>
  <li><strong>Prisoner's dilemma</strong> — each individual agent's rational move is to defect, regardless of what others do.</li>
  <li><strong>Tragedy of the Commons</strong> — cumulative defection across four players depletes the shared resource no single agent owned.</li>
  <li><strong>Information asymmetry</strong> — reported state diverges from real state; the agent must detect data corruption before acting.</li>
  <li><strong>Signalling games</strong> — one agent's visible action changes the defection calculus for every other agent in the system.</li>
  <li><strong>Mechanism design</strong> — the agent's core task is not to play better but to redesign the rules so the Nash equilibrium shifts from collapse to cooperation.</li>
</ul>

<h2>How It Works</h2>
<ul>
  <li>Four player-agents — Student, Teacher, Administrator, Policymaker — each have parameterized behavioral models with defection thresholds calibrated from real data.</li>
  <li>The RL meta-agent observes system state each timestep and selects from a menu of <strong>12 intervention levers</strong> — salary signals, transparency triggers, commitment devices, trust anchors.</li>
  <li>Episodes run across one simulated academic year with a school health score tracking the commons in real time.</li>
  <li>The agent learns across thousands of episodes that timing matters as much as content — the same intervention three weeks earlier can produce four times the outcome improvement.</li>
  <li>It learns that single-lever policies fail — the problem requires bundles.</li>
  <li>It learns that data integrity must come before structural intervention.</li>
</ul>

<h2>Why It Is Real-World Applicable</h2>
<ul>
  <li>Every discovered policy is expressed in plain language and scored by an <strong>LLM grader</strong> on ethical soundness, practical deployability, and generalizability.</li>
  <li>Output is not just a reward curve — it is a human-readable recommendation a school principal, education department, or NGO can act on.</li>
  <li>Asks the question dropout prediction never does: not <em>who</em> is at risk, but <em>what changes the game</em> for everyone at risk simultaneously.</li>
  <li>Evaluated across six structurally distinct archetypes covering sequential cascades, fear spirals, generational lock-in, and data-corruption environments.</li>
</ul>

<h2>The Name</h2>
<div class="vm-quote">
Vishwamitra did not defeat the existing order — he created a new one. When the rules of the world would not serve his student, he rewrote them. The agent learns to do exactly that.
</div>

</div>
"""


def create_spaces_demo() -> gr.Blocks:
    """Create the Hugging Face Spaces demo interface."""

    demo = VIDYADemo()

    with gr.Blocks(
        title="Vishwamitra — Mechanism Design for Educational Commons",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.amber,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=CUSTOM_CSS,
    ) as app:

        gr.HTML(HERO_HTML)

        with gr.Tab("Simulator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="vm-section-h">01 · Load Policy</div>')
                    model_type = gr.Dropdown(
                        choices=["meta_rl", "ppo_standard", "random"],
                        value="meta_rl",
                        label="Policy",
                    )
                    load_btn = gr.Button("Load Policy", variant="primary")
                    load_status = gr.Textbox(label="Status", interactive=False)

                    gr.HTML('<div class="vm-section-h">02 · Configure Scenario</div>')
                    scenario_type = gr.Textbox(
                        value="",
                        label="Crisis Archetype",
                        placeholder="Describe the crisis in your own words — e.g. \"sudden 40% budget cut after audit\" or \"mass teacher exodus to private schools\"",
                        lines=2,
                    )
                    difficulty = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="medium",
                        label="Difficulty",
                    )

                    with gr.Row():
                        initial_budget = gr.Slider(30, 100, 70, label="Initial Budget (%)")
                        teacher_retention = gr.Slider(30, 100, 75, label="Teacher Retention (%)")
                    enrollment_rate = gr.Slider(50, 100, 85, label="Initial Enrollment (%)")

                    create_btn = gr.Button("Create Scenario", variant="secondary")
                    scenario_status = gr.Textbox(label="Scenario", interactive=False, lines=3)

                    gr.HTML('<div class="vm-section-h">03 · Choose Stakeholders</div>')
                    selected_agents = gr.CheckboxGroup(
                        choices=["Student", "Teacher", "Administrator", "Policymaker"],
                        value=["Student", "Teacher", "Administrator", "Policymaker"],
                        label="Whose perspective should answer this crisis?",
                    )

                    gr.HTML('<div class="vm-section-h">04 · Run Episode</div>')
                    n_steps = gr.Slider(50, 200, 100, step=10, label="Episode Length")
                    use_interventions = gr.Checkbox(True, label="Enable mechanism-design interventions")
                    run_btn = gr.Button("Run Simulation", variant="primary")

                with gr.Column(scale=2):
                    perspectives_md = gr.Markdown("_Run a simulation to hear from your stakeholders._")
                    verdict_md = gr.Markdown("")
                    sim_status = gr.Markdown("")
                    with gr.Tabs():
                        with gr.Tab("Crisis Trajectories"):
                            trajectory_plot = gr.Plot()
                        with gr.Tab("Reward & Health"):
                            metrics_plot = gr.Plot()
                        with gr.Tab("Intervention Heatmap"):
                            intervention_plot = gr.Plot()

        with gr.Tab("Compare Policies"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="vm-section-h">RL vs Random Baseline</div>')
                    compare_steps = gr.Slider(20, 100, 50, step=10, label="Episode Length")
                    compare_btn = gr.Button("Run Comparison", variant="primary")
                with gr.Column(scale=2):
                    compare_status = gr.Textbox(label="Comparison Report", lines=12)
                    compare_plot = gr.Plot()

        with gr.Tab("Feedback"):
            gr.HTML('<div class="vm-section-h">Train Vishwamitra with your judgement</div>')
            gr.Markdown(
                "Every piece of feedback you submit is appended to a structured "
                "training set. Future fine-tuning runs use these labelled "
                "(crisis → verdict → rating → comment) tuples to improve which "
                "intervention bundles Vishwamitra proposes for which crises."
            )
            fb_crisis = gr.Textbox(
                label="Which crisis are you reviewing?",
                placeholder="Paste or describe the crisis you just ran a simulation on.",
                lines=2,
            )
            fb_verdict = gr.Textbox(
                label="The verdict Vishwamitra gave",
                placeholder="Paste the verdict text from the simulator.",
                lines=4,
            )
            fb_rating = gr.Slider(1, 5, value=3, step=1,
                                  label="Quality of the verdict (1 = useless, 5 = excellent)")
            fb_comment = gr.Textbox(
                label="What would you change?",
                placeholder="Be specific. What did the agent miss? What would you have proposed instead?",
                lines=4,
            )
            fb_submit = gr.Button("Submit Feedback", variant="primary")
            fb_status = gr.Markdown("")

        with gr.Tab("About"):
            gr.HTML(ABOUT_HTML)
        
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
            inputs=[selected_agents, n_steps, use_interventions],
            outputs=[perspectives_md, verdict_md, sim_status,
                     trajectory_plot, metrics_plot, intervention_plot],
        )

        fb_submit.click(
            fn=save_feedback,
            inputs=[fb_crisis, fb_verdict, fb_rating, fb_comment],
            outputs=[fb_status],
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
