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
