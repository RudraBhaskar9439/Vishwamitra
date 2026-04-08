---
title: Vishwamitra — Mechanism Design for Educational Commons
emoji: 🪔
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
---

<div align="center">

# 🪔 Vishwamitra

### *Seeing clearly. Redesigning the game.*

**A multi-agent reinforcement learning environment that learns to rewrite the rules of educational collapse — before it happens.**

[![Built on Meta OpenEnv](https://img.shields.io/badge/Built_on-Meta_OpenEnv-0668E1?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/openenv)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-008080?style=for-the-badge)](https://gymnasium.farama.org)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/rudra9439/vidya-meta-rl)
[![Meta · PyTorch Hackathon](https://img.shields.io/badge/Meta_·_PyTorch_Hackathon-OpenEnv_Round_1-blueviolet?style=for-the-badge)](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**[🚀 Live Demo](https://huggingface.co/spaces/rudra9439/vidya-meta-rl) · [📄 Submission Spec](#-openenv-round-1-submission) · [🧠 How It Works](#-how-it-works) · [🎮 Try a Crisis](#-try-it-yourself)**

</div>

---

## ✨ The One-Sentence Pitch

> **Public education doesn't fail because of bad people — it fails because four rational stakeholders, each playing their dominant strategy, collectively produce collapse. Vishwamitra is an RL meta-agent that detects the cascade early and rewrites the incentive structure so cooperation becomes the dominant strategy for everyone.**

---

## 🌍 The Problem

Every stakeholder in a failing school makes a locally rational choice:

| Stakeholder | Rational defection | What they don't see |
|---|---|---|
| 🎓 **Student** | Skip class — there's no point | Their absence accelerates the cascade |
| 👨‍🏫 **Teacher** | Burn out and quit | Their exit forces the next teacher to defect |
| 🏛️ **Administrator** | Delay hard decisions | Delay compounds the rumour mill |
| 🏢 **Policymaker** | Redirect funds to visible wins | Underinvestment creates the next crisis |

**None of them chose collapse. All of them, acting independently, produce it together.**

This is the **Tragedy of the Commons** applied to education. It is a **game theory** problem, not a resource problem. And game theory problems have game-theoretic solutions.

---

## 🏛️ Architecture

> 📐 **Architecture diagram placeholder** — drop your final diagram in `docs/architecture.png` and reference it below.

<!-- ![Vishwamitra Architecture](docs/architecture.png) -->

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER (you)                                 │
│   types a free-text crisis  +  picks stakeholders to consult        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                ┌────────────────┴─────────────────┐
                ▼                                  ▼
   ┌─────────────────────────┐      ┌──────────────────────────────┐
   │   DropoutCommonsEnv     │      │   LLM Advisory Layer         │
   │  (Gymnasium / OpenEnv)  │      │  (OpenAI-compatible client)  │
   │                         │      │                              │
   │  • 4 simulated agents   │      │  • Stakeholder personas      │
   │  • 13-d obs / 8-d act   │      │  • Mechanism-design verdict  │
   │  • Dense reward         │      │  • Dynamic plot curator      │
   │  • Scenario shocks      │      │  • Retrieval-aug from        │
   │                         │      │    feedback.jsonl            │
   └───────────┬─────────────┘      └────────────┬─────────────────┘
               │ trajectory                       │ verdict + perspectives
               └──────────────┬───────────────────┘
                              ▼
              ┌───────────────────────────────────┐
              │      Vishwamitra UI (Gradio)      │
              │   • Dynamic crisis trajectories   │
              │   • Stakeholder voices            │
              │   • Final verdict                 │
              │   • Inline feedback → retry loop  │
              └────────────────┬──────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   feedback.jsonl (memory)    │
                │   ─────────────────────────  │
                │   Every rated verdict is     │
                │   retrieved on the next      │
                │   similar crisis and         │
                │   injected as a lesson       │
                └──────────────────────────────┘
```

---

## 🧠 How It Works

### 1. The Simulator — `DropoutCommonsEnv`

A Gymnasium-compatible multi-agent environment built on Meta's OpenEnv contract.

- **Observation space**: `Box(13,)` — 13 normalized health metrics (enrollment, dropout, retention, burnout, engagement, budget, etc.)
- **Action space**: `Box(8,)` — 8 continuous intervention intensities (funding boost, teacher incentives, scholarships, attendance mandates, resource reallocation, transparency reports, staff hiring, counseling)
- **Reward**: dense, with partial-progress signals — combines dropout penalty + retention bonus + engagement bonus − intervention cost
- **Termination**: early collapse on dropout > 50%, retention < 20%, enrollment < 30%, or budget exhausted
- **Stakeholders**: four parameterized behavioral models — Student, Teacher, Administrator, Policymaker — each with defection thresholds calibrated from real data

### 2. The LLM Advisory Layer

When you run a simulation, the env trajectory is paired with LLM reasoning:

- Each selected stakeholder gets a **persona prompt** and produces a 4–6 sentence first-person reaction
- Vishwamitra synthesizes a **final verdict** in a strict 4-part structure: Diagnosis → Intervention Bundle → Why this shifts the equilibrium → First action this week
- A separate LLM call **dynamically curates** which 4 of the 13 metrics to chart and what to title each plot — so the visualization adapts to *your* crisis, not a fixed template

### 3. The Learning Loop

Below the Run Simulation button there's an inline feedback section:

```
Quality slider (1–5)  +  "What would you change?"  →  Submit Feedback & Retry
```

When you submit:

1. The `(crisis, verdict, rating, comment)` tuple is appended to `feedback.jsonl`
2. The same scenario re-runs **immediately** with that feedback injected as a fresh lesson
3. On every future Run Simulation, a **Jaccard token retriever** pulls the top-3 most relevant past lessons from the feedback store
4. High-rated past verdicts are labeled `✅ HIGHLY RATED` → reused
5. Low-rated past verdicts are labeled `⚠️ POORLY RATED — DO NOT REPEAT` → avoided
6. User correction comments are pasted in verbatim with explicit instructions to incorporate them

This is **retrieval-augmented in-context learning**, not gradient-based RL fine-tuning. The system observably improves on similar future crises after feedback is submitted, via persistent retrieval from a growing feedback store. No GPU, no training loop, no waiting — the next answer is informed by the last lesson.

---

## 🎲 The Game Theory Stack

| Concept | Where it shows up in Vishwamitra |
|---|---|
| **Prisoner's Dilemma** | Each stakeholder's individually rational move is to defect — skip class, quit, delay, redirect funds |
| **Tragedy of the Commons** | Cumulative defection across all four players depletes a shared resource no single agent owned |
| **Information Asymmetry** | Reported state diverges from real state — the agent must reason about data corruption before acting |
| **Signalling Games** | One agent's visible action shifts the defection calculus for every other agent |
| **Mechanism Design** | The agent's job is not to play better but to **redesign the rules** so the Nash equilibrium shifts from collapse to cooperation |

---

## 🚀 Try It Yourself

### On the live Space

1. Open the [**🚀 Live Demo**](https://huggingface.co/spaces/rudra9439/vidya-meta-rl)
2. Click **Load Policy**
3. In **Configure Scenario**, type any crisis you can imagine — e.g.:
   > *"The headmistress of a girls' high school in Jharkhand has been quietly absent for six weeks. Three teachers haven't been paid in two months. Class 9 and 10 girls are dropping out fastest because their families need them home during harvest season. WhatsApp groups are full of conflicting rumours."*
4. Pick which stakeholders should weigh in
5. Click **Run Simulation**
6. Read the four perspectives + the verdict
7. Below the Run button, rate the verdict and tell Vishwamitra what to do better → click **Submit Feedback & Retry**
8. Watch the verdict update in place with your correction baked in

### Locally

```bash
git clone https://github.com/<you>/vishwamitra.git
cd vishwamitra
pip install -r requirements.txt

# Set up your LLM endpoint (Groq is fast & free)
cat > .env <<EOF
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
HF_TOKEN=gsk_your_groq_key
EOF

python app.py             # Gradio UI on http://localhost:7860
python inference.py       # Headless OpenEnv-compliant evaluation
python validate.py        # Pre-submission self-checks
```

---

## 📐 Environment Specification

### Observation Space — `Box(low=0, high=1, shape=(13,), dtype=float32)`

| Index | Field | Meaning |
|---|---|---|
| 0 | `enrollment_rate` | Fraction of school-age population enrolled |
| 1 | `attendance_rate` | Fraction of enrolled students attending |
| 2 | `dropout_rate` | Fraction of enrolled students dropping out |
| 3 | `teacher_retention` | Fraction of teachers retained year-over-year |
| 4 | `budget_utilization` | Fraction of allocated budget actually spent |
| 5 | `class_size_norm` | Average class size, normalized to 60 |
| 6 | `teacher_workload` | Aggregate workload index |
| 7 | `resource_allocation` | Quality of resource distribution |
| 8 | `student_engagement` | Composite engagement signal |
| 9 | `teacher_burnout` | Burnout index (lower is better) |
| 10 | `policy_compliance` | Compliance with district policy |
| 11 | `budget_remaining_norm` | Remaining budget, normalized to 2M |
| 12 | `step` | Time step within the episode |

### Action Space — `Box(low=0, high=1, shape=(8,), dtype=float32)`

| Index | Lever | Per-step cost (units) |
|---|---|---|
| 0 | `funding_boost` | 50,000 |
| 1 | `teacher_incentive` | 80,000 |
| 2 | `student_scholarship` | 30,000 |
| 3 | `attendance_mandate` | 10,000 |
| 4 | `resource_realloc` | 40,000 |
| 5 | `transparency_report` | 5,000 |
| 6 | `staff_hiring` | 120,000 |
| 7 | `counseling_programs` | 25,000 |

### Tasks (agent-graded)

| ID | Difficulty | Scenario | Episode Length | Pass Threshold |
|---|---|---|---|---|
| `task_easy_funding` | easy | funding_cut | 40 | avg health ≥ 0.55 |
| `task_medium_teacher_shortage` | medium | teacher_shortage | 60 | avg health ≥ 0.50 |
| `task_hard_pandemic` | hard | pandemic_recovery | 80 | avg health ≥ 0.48 |

---

## 🏆 OpenEnv Round 1 Submission

This repository is a **complete, validated submission** for the Meta · PyTorch Hackathon OpenEnv Round 1.

| Requirement | Artifact | Status |
|---|---|---|
| `inference.py` at root, `[START]/[STEP]/[END]` log format | [`inference.py`](inference.py) | ✅ |
| `openenv.yaml` env + task manifest | [`openenv.yaml`](openenv.yaml) | ✅ |
| `Dockerfile` builds successfully | [`Dockerfile`](Dockerfile) | ✅ |
| 3+ tasks easy / medium / hard with agent graders | [`inference.py`](inference.py) | ✅ |
| Reward function with partial progress signals | [`env/dropout_env.py`](env/dropout_env.py) | ✅ |
| OpenAI client + `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN` | [`inference.py`](inference.py) | ✅ |
| Runtime < 20 min on 2 vCPU / 8 GB | ~3–6 min on Groq | ✅ |
| HF Space deployed and reachable | [Space link](https://huggingface.co/spaces/rudra9439/vidya-meta-rl) | ✅ |
| Pre-submission self-validator | [`validate.py`](validate.py) | ✅ 20/20 |
| Real-world task (not toy/game) | Education systems | ✅ |

Run the validator yourself:

```bash
python validate.py
```

---

## 🗂️ Project Structure

```
vidya-meta-rl/
├── app.py                    # Gradio UI — Vishwamitra advisory experience
├── inference.py              # OpenEnv submission entry point
├── openenv.yaml              # OpenEnv environment + task manifest
├── Dockerfile                # Reproducible container
├── validate.py               # Pre-submission self-check (20/20)
├── requirements.txt          # Python deps
│
├── env/                      # The simulator
│   ├── dropout_env.py        # DropoutCommonsEnv (Gymnasium API)
│   ├── state.py              # SystemState dataclass + obs encoding
│   ├── spaces.py             # Observation/action space factories
│   ├── openenv_compat.py     # OpenEnv wrapper
│   ├── collapse_detector.py  # Early-warning collapse heuristics
│   └── scenarios/            # Crisis archetypes
│       ├── funding_cut.py
│       ├── teacher_shortage.py
│       ├── pandemic_recovery.py
│       ├── conflict_zone.py
│       └── indian_context.py
│
├── agents/                   # Behavioural models
│   ├── student_agent.py
│   ├── teacher_agent.py
│   ├── admin_agent.py
│   ├── policymaker_agent.py
│   └── adversarial_agent.py  # Stress-test adversary
│
├── training/                 # Offline RL training infrastructure
│   ├── train.py              # PPO baseline
│   ├── meta_rl.py            # MAML meta-policy
│   ├── curriculum.py         # Curriculum learning
│   └── callbacks.py
│
├── checkpoints/              # Trained policy weights
└── docs/                     # (placeholder for diagrams)
```

---

## 🛠️ Tech Stack

- **PyTorch** — meta-RL training (MAML / PPO via Stable-Baselines3)
- **Gymnasium** — environment contract
- **Meta OpenEnv** — multi-environment integration spec
- **Plotly + Gradio** — interactive UI on Hugging Face Spaces
- **OpenAI-compatible client** — works with Groq, Fireworks, Together, HF Inference, Cerebras, Ollama
- **python-dotenv** — local credential management
- **Docker** — reproducible deployment

---

## 🎯 What Makes This Different

Most "AI for education" projects ask: *which student is at risk?*
**Vishwamitra asks the harder question: what changes the game for everyone at risk simultaneously?**

- ✅ **Game-theoretic framing**, not predictive modelling
- ✅ **Multi-stakeholder reasoning** — four distinct LLM personas, not one-voice oracle
- ✅ **Dynamic visualizations** — chart titles & metric selection adapt to your specific crisis
- ✅ **Inline learning loop** — feedback updates the next answer immediately, not "in a future fine-tune"
- ✅ **OpenEnv-native** — runs as a headless inference target *and* as an interactive Space
- ✅ **Calibrated simulator** — the env behind the LLM is a real Gymnasium environment with dense rewards and collapse dynamics, not a toy
- ✅ **Plain-language deployable verdicts** — the output is something a principal, a district officer, or an NGO can actually act on this week

---

## 🪔 The Name

> Vishwamitra did not defeat the existing order — he created a new one.
> When the rules of the world would not serve his student, he rewrote them.
> The agent learns to do exactly that.

---

## 📜 License

MIT — see [`LICENSE`](LICENSE).

## 🙏 Acknowledgements

Built for the **Meta · PyTorch Hackathon — OpenEnv Round 1**, organised by Scaler School of Technology.

Powered by Meta's [OpenEnv](https://github.com/facebookresearch/openenv) framework and the open-source PyTorch / Gymnasium / Hugging Face ecosystem.

## 👤 Author

**Rudra Bhaskar**
[Hugging Face](https://huggingface.co/rudra9439) · [GitHub](https://github.com/rudra9439)

---

<div align="center">

**If Vishwamitra helped you think differently about a coordination failure, ⭐ star the repo.**

*Seeing clearly. Redesigning the game.*

</div>
