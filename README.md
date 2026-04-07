---
title: VIDYA Educational Crisis Simulator
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# VIDYA: Educational System Crisis Simulator

Interactive demo for testing AI policies on educational crisis scenarios using Meta-RL (MAML).

## Features

- 🎮 **Interactive Simulations**: Test different crisis scenarios
- 🤖 **Meta-RL Policies**: Policies that adapt to new scenarios
- 📊 **Real-time Visualization**: Track system metrics and interventions
- ⚖️ **Policy Comparison**: Compare RL vs random baseline

## Scenarios

- **Funding Crisis**: Sudden budget cuts affecting school operations
- **Teacher Shortage**: Mass teacher departures creating staffing crisis
- **Pandemic Recovery**: Post-pandemic learning loss and enrollment challenges
- **Conflict Zone**: Education under protracted armed conflict

## Interventions

The AI can deploy 8 types of interventions:
1. 💰 Funding boost
2. 👨‍🏫 Teacher incentives
3. 🎓 Student scholarships
4. 📊 Attendance mandates
5. 🔄 Resource reallocation
6. 📢 Transparency reports
7. 👥 Staff hiring
8. 💬 Counseling programs

## Meta-Learning (MAML)

This demo uses Model-Agnostic Meta-Learning to train policies that can quickly adapt to new crisis scenarios with just a few gradient steps.

## Environment Description (OpenEnv)

VIDYA's `DropoutCommonsEnv` is a Gymnasium-compatible environment that
simulates the dynamics of an educational system under stress. The agent
acts as a meta-policy maker selecting intervention intensities each step;
internal simulated agents (student / teacher / admin / policymaker) react,
and scenario-specific shocks (funding cuts, teacher exodus, pandemic
recovery, etc.) perturb the system.

### Observation Space
`Box(low=0, high=1, shape=(13,), dtype=float32)` — normalized metrics:
`enrollment_rate, attendance_rate, dropout_rate, teacher_retention,
budget_utilization, class_size_norm, teacher_workload, resource_allocation,
student_engagement, teacher_burnout, policy_compliance,
budget_remaining_norm, step`.

### Action Space
`Box(low=0, high=1, shape=(8,), dtype=float32)` — intervention intensities:
1. funding_boost
2. teacher_incentive
3. student_scholarship
4. attendance_mandate
5. resource_realloc
6. transparency_report
7. staff_hiring
8. counseling_programs

### Reward
Dense reward combining a dropout penalty, teacher-retention bonus,
student-engagement bonus, and an intervention-cost penalty. Provides a
partial-progress signal at every step. The episode terminates early on
*system collapse* (dropout > 50%, teacher retention < 20%, budget
exhausted, or enrollment < 30%).

### Tasks (agent-graded)
| ID | Difficulty | Scenario | Episode | Threshold |
|---|---|---|---|---|
| `task_easy_funding` | easy | funding_cut | 40 | avg health ≥ 0.55 |
| `task_medium_teacher_shortage` | medium | teacher_shortage | 60 | avg health ≥ 0.50 |
| `task_hard_pandemic` | hard | pandemic_recovery | 80 | avg health ≥ 0.48 |

## OpenEnv Submission

- `inference.py` — entry point. Runs all 3 tasks with an LLM policy via
  the OpenAI-compatible client. Required env vars:
  `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`. Emits structured
  `[START] / [STEP] / [END]` logs to stdout. Fits 2 vCPU / 8 GB and
  finishes in well under 20 minutes.
- `openenv.yaml` — environment + task manifest.
- `Dockerfile` — reproducible container; default CMD runs `inference.py`.
- `validate.py` — pre-submission self-checks (`python validate.py`).

Run inference locally:
```bash
export API_BASE_URL=https://your-endpoint
export MODEL_NAME=your-model
export HF_TOKEN=hf_xxx
python inference.py
```

Build & run the container:
```bash
docker build -t vidya .
docker run --rm \
  -e API_BASE_URL -e MODEL_NAME -e HF_TOKEN \
  vidya
```

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:7860` to access the demo.

## Citation

```bibtex
@software{vidya,
  title = {VIDYA: Educational System Crisis Simulator},
  author = {Rudra Bhaskar},
  year = {2024},
  url = {https://huggingface.co/spaces/your-username/vidya}
}
```
