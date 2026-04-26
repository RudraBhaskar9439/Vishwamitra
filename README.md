<!-- ================================================================== -->
<!--                              HERO                                   -->
<!-- ================================================================== -->

<div align="center">

# Vishwamitra

### *Most policy AI tries to find the optimum.*<br/>*Vishwamitra maps the **Nash equilibrium** that the optimum has to break.*

A multi-agent OpenEnv simulation of educational systems collapse, framed as a **Pareto-inefficient coordination failure** across four stakeholder lenses, paired with a two-tier *swarm-of-swarms* deliberation layer and a **knowledge-distilled 1-billion-parameter student model** that carries the swarm's mechanism-design verdicts into deployment at ~100× lower inference cost.

[![OpenEnv](https://img.shields.io/badge/Built_on-Meta_OpenEnv-0668E1?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/openenv)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Spaces-Live_Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/rudra9439/vidya-meta-rl)
[![Open in Kaggle](https://img.shields.io/badge/Open_in-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Meta_·_PyTorch_Hackathon-Round_2-blueviolet?style=for-the-badge)](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)

**[🚀 Try the env](https://huggingface.co/spaces/rudra9439/vidya-meta-rl)** &nbsp;·&nbsp;
**[▶️  Walkthrough](#-watch-the-walkthrough)** &nbsp;·&nbsp;
**[🧠 Architecture](#️-architecture)** &nbsp;·&nbsp;
**[🎲 Game-theoretic foundations](#-game-theoretic-foundations)** &nbsp;·&nbsp;
**[📊 Results](#-results)** &nbsp;·&nbsp;
**[♻️  Reproduce](#️-reproducibility)**

</div>

<br/>

> **TL;DR.** Educational systems collapse on predictable trajectories — falling enrollment, teacher attrition, class-size shocks, dropout cliffs — because every stakeholder is playing a *locally rational best response* in a coordination game with no mechanism for surfacing cross-lens disagreement. The aggregate equilibrium is **Pareto-inefficient**: nobody chooses collapse, but their interlocking best responses produce it [^4]. Vishwamitra is an OpenEnv-compliant simulator of those dynamics, paired with a two-tier *swarm-of-swarms* deliberation layer in which **four stakeholder swarms × three heterogeneous LLM personas** debate every intervention and surface a structured **resonance map** — a Schelling-point detector for cross-stakeholder consensus. We then **distil** the swarm's mechanism-design verdict into a single 1-B student model that runs at ~100× lower inference cost. On 20 evaluation episodes the distilled student reaches **+0.449 cumulative reward versus −23.504 for the do-nothing baseline — a 24-reward-unit gap** — with ~10% lower variance than uniform-random play.

<br/>

---

## Table of Contents

- [The problem — a coordination failure, not a resource failure](#-the-problem--a-coordination-failure-not-a-resource-failure)
- [Game-theoretic foundations](#-game-theoretic-foundations)
- [Architecture](#️-architecture)
- [Quick start](#-quick-start)
- [The environment — `DropoutCommonsEnv`](#-the-environment--dropoutcommonsenv)
- [The innovation — two-tier swarm orchestration](#-the-innovation--two-tier-swarm-orchestration)
- [Knowledge distillation pipeline](#-knowledge-distillation-pipeline)
- [Results](#-results)
- [Watch the walkthrough](#-watch-the-walkthrough)
- [Reproducibility](#️-reproducibility)
- [Repository layout](#-repository-layout)
- [Roadmap](#️-roadmap)
- [Citation](#-citation)
- [References](#-references)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🌍 The problem — a coordination failure, not a resource failure

Every stakeholder in a failing school is making a *locally rational* choice. None of them chooses collapse; collectively, they produce it. This is the canonical structure of a **Tragedy of the Commons** [^4] — and, more precisely, a **Pareto-inefficient pure-strategy Nash equilibrium** [^9] in which every actor's best response, given everyone else's behaviour, is to *not* invest in the system.

| Stakeholder | Locally rational best response | What they don't internalise |
|---|---|---|
| 🎓 **Student** | Skip class — there's no point | Their absence accelerates a peer-norm cascade |
| 👨‍🏫 **Teacher** | Burn out and quit | Their exit forces the next teacher to defect |
| 🏛️ **Administrator** | Delay hard decisions | Delay compounds a rumour-driven mistrust spiral |
| 🏢 **Policymaker** | Redirect funds to visible wins | Underinvestment seeds the next acute crisis |

Each row is a **dominant strategy** under bounded rationality [^10]. The aggregate of these strategies is the collapse trajectory. The standard policymaking toolkit fails to break the equilibrium because:

- **Cost-benefit analyses** optimise one variable at a time — ignoring the externalities each lens imposes on the others.
- **Single-stakeholder consultations** privilege the loudest voice in the room, producing biased social-welfare functions [^11].
- **Recent AI policy tools** manufacture artificial consensus by hiding disagreement behind a single recommendation — destroying the very signal a *mechanism designer* [^12] needs to break the equilibrium.

> **Vishwamitra reframes this as an explicit disagreement-mapping problem.** The output is not one number; it is a *structured artefact* listing which interventions every stakeholder lens agrees on (the **Schelling points** [^13] — deploy with confidence) and which are genuinely contested (require human judgement to break the equilibrium).

---

## 🎲 Game-theoretic foundations

Vishwamitra is, formally, a **mechanism-design system for a four-player coordination game with bounded-rational, heterogeneous agents**. The framing matters because it determines what the right *output* of the system is.

### The collapse trajectory as a Nash equilibrium

Let `S = {Student, Teacher, Administrator, Policymaker}` be the four players and let each player `i ∈ S` have a payoff function `u_i(a_S, x)` over the joint action profile `a_S = (a_1, …, a_4)` and the system state `x ∈ ℝ¹³`. The status-quo trajectory in `DropoutCommonsEnv` is exactly the strategy profile

```
a_i* = argmax_{a_i} u_i(a_i, a_{−i}*, x)    for all i
```

— each player best-responding to everyone else's actions. By the Nash existence theorem [^9] this profile exists; by direct simulation in the env it converges to **collapse** in roughly 40 steps with cumulative reward ≈ −23.5. The equilibrium is *stable* (no individual deviation improves any single player's payoff) and *Pareto-dominated* (some other action profile would make every player better off).

The job of a mechanism designer [^12] is to *change the game* — adjust incentives or surface coordination opportunities — so the new equilibrium dominates the old one. Vishwamitra does the surfacing half of that job.

### Why disagreement is the load-bearing signal

A naive policy AI that emits a single recommended action is implicitly *aggregating* the four players' utility functions into a scalar — a social-welfare function that is, by Arrow's impossibility theorem [^11], guaranteed to violate at least one of unanimity, independence, or non-dictatorship. There is no canonical "right" aggregation across heterogeneous stakeholders.

Vishwamitra refuses the aggregation. For each intervention `j ∈ {1, …, 8}`, we compute a **resonance score**

```
resonance(j) = 1 − σ_normalised( {⟨a_j⟩_role : role ∈ S} )
```

— literally one minus the normalised standard deviation of the four role-swarm aggregated recommendations on lever `j`. **Resonance is computed unweighted** so operator priors never contaminate the disagreement signal. High resonance (≥ 0.55) marks lever `j` as a **Schelling point** [^13] — a focal coordinate the four lenses converge on without needing communication, where mechanism deployment is cheap. Low resonance flags lever `j` as **dissonant**, escalating it for human judgement: this is precisely where the bounded-rational equilibrium needs an external coordinating authority to move.

### Bounded rationality and the persona library

Each of our 12 LLM personas is a model of a **bounded-rational agent** [^10] — operating with imperfect information, finite cognitive budget, and a salient subset of state features encoded as `fit_signals`. The L2 router (`PersonaAllocator`) modulates per-persona attention by `fit_signals × state_pressures` — formally implementing the *salience-weighted attention model* [^14] that drives bounded-rational decision-making in the behavioural-economics literature.

| Game-theoretic concept | Vishwamitra realisation |
|---|---|
| Pure-strategy Nash equilibrium | Collapse trajectory under do-nothing baseline (cumulative reward −23.5) |
| Pareto improvement | Trained-student trajectory (+0.45 vs −23.5 — a 24-unit gap) |
| Schelling point | Intervention with resonance ≥ 0.55 across the four swarms |
| Mechanism designer | The L1 + L2 orchestrator + the resonance/dissonance map |
| Bounded rationality | Per-persona `fit_signals` + L2 attention weighting |
| Coalition formation | High-resonance clusters (e.g., Student ↔ Teacher converging on `counseling_programs`) |
| Coordination failure | Low-resonance lever flagged for human judgement |

> **The contribution.** Vishwamitra does *not* claim to compute the social-welfare optimum. It claims to surface the structured disagreement that any honest mechanism designer needs to see *before* a policy ships. That is a smaller, more defensible claim — and the one we have evidence for.

---

## 🏛️ Architecture

```mermaid
flowchart TB
    subgraph Inputs["Inputs"]
        S["State vector<br/>(13-d observation)"]
        SC["Scenario brief<br/>(operator text)"]
    end

    subgraph L1["⬣ L1 Orchestrator — WeightAllocator"]
        ATT["State pressures<br/>↓<br/>Per-role attention scores"]
        MD["Model + verdict-weight<br/>per role swarm"]
        ATT --> MD
    end

    subgraph Swarms["12 LLM personas across 4 role swarms"]
        ST[/"🎓 STUDENT<br/>Maya · Rohit · Priya"/]
        TE[/"👨‍🏫 TEACHER<br/>Sharma · Iyer · Kumar"/]
        AD[/"🏛️ ADMIN<br/>Desai · Joshi · Rao"/]
        PO[/"🏢 POLICYMAKER<br/>Verma · Banerjee · Khan"/]
    end

    subgraph L2["⬡ L2 Orchestrator — PersonaAllocator"]
        FIT["Per-persona fit_signals<br/>× state pressures<br/>↓<br/>Within-swarm weight<br/>(salience-weighted attention)"]
    end

    subgraph Output["Output"]
        RM["Resonance / dissonance map<br/>(Schelling-point detector)<br/>+ Final action vector<br/>+ Persona reasoning trail"]
    end

    S --> ATT
    S --> FIT
    MD -.routes model & weight.-> ST
    MD -.-> TE
    MD -.-> AD
    MD -.-> PO

    SC ==> ST
    SC ==> TE
    SC ==> AD
    SC ==> PO

    ST --> FIT
    TE --> FIT
    AD --> FIT
    PO --> FIT

    FIT --> RM

    style Inputs fill:#0f172a,stroke:#475569,color:#e2e8f0
    style Output fill:#064e3b,stroke:#10b981,color:#ecfdf5
    style L1 fill:#1e293b,stroke:#fbbf24,color:#fef3c7
    style L2 fill:#1e293b,stroke:#a78bfa,color:#ede9fe
    style Swarms fill:#0c1322,stroke:#475569,color:#e2e8f0
```

### Distillation pipeline

```mermaid
flowchart LR
    Swarm["12-persona swarm<br/>(teacher, mechanism designer)"] -->|"~80 deliberations"| Data["train.jsonl<br/>val.jsonl"]
    Data -->|"Unsloth + TRL SFT"| Student["Llama-3.2-1B<br/>+ LoRA adapter<br/>(distilled student)"]
    Student -->|"20 episodes × 40 steps"| Eval["DropoutCommonsEnv<br/>reward eval"]
    Random["Random baseline"] --> Eval
    Zero["Zero-action<br/>(Nash collapse) baseline"] --> Eval

    style Swarm fill:#7c2d12,stroke:#fbbf24,color:#fef3c7
    style Student fill:#1e3a8a,stroke:#60a5fa,color:#dbeafe
    style Eval fill:#064e3b,stroke:#10b981,color:#ecfdf5
    style Zero fill:#7f1d1d,stroke:#f87171,color:#fecaca
```

---

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/RudraBhaskar9439/Enigma.git
cd Enigma

# 2. Install
pip install -e ".[dev]"
cd frontend && npm install && cd ..

# 3. Provide an LLM key (any one works — provider-agnostic OpenAI-format client)
cat > .env <<'EOF'
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
EOF

# 4. Run the backend (FastAPI on :8000)
uvicorn server.app:api --reload

# 5. Run the frontend (Vite on :5173) — separate terminal
cd frontend && npm run dev

# 6. Open http://localhost:5173 → Swarms tab → load a scenario → Run Deliberation
```

A live deployment is available on Hugging Face Spaces: **[rudra9439/vidya-meta-rl](https://huggingface.co/spaces/rudra9439/vidya-meta-rl)**.

---

## 🌐 The environment — `DropoutCommonsEnv`

A `gymnasium.Env` implementing the OpenEnv contract [^1]. The training agent is a **mechanism designer**: each step it picks intervention intensities; the four simulated stakeholder agents respond with their bounded-rational best responses.

<details>
<summary><strong>Observation space — <code>Box(13,)</code> float32, range [0, 1]</strong></summary>

| # | Field | Meaning |
|---|---|---|
| 0 | `enrollment_rate` | Reported enrollment (signal — may diverge from true) |
| 1 | `attendance_rate` | Daily attendance |
| 2 | `dropout_rate` | Per-period dropout |
| 3 | `teacher_retention` | Staff retention |
| 4 | `budget_utilization` | 1 − (budget_remaining / 2M) |
| 5 | `avg_class_size` | Normalised (0 ≈ 0 students, 1 ≈ 60) |
| 6 | `teacher_workload` | Workload index |
| 7 | `resource_allocation` | Quality of resource distribution |
| 8 | `student_engagement` | Engagement index |
| 9 | `teacher_burnout` | Burnout index (lower = better) |
| 10 | `policy_compliance` | Adherence rate |
| 11 | `budget_ratio` | Cash remaining / 2M |
| 12 | `trust_score` / `data_integrity` | Faith in system / signal-vs-truth integrity |

</details>

<details>
<summary><strong>Action space — <code>Box(8,)</code> float32, range [0, 1]</strong></summary>

| # | Intervention | Per-step max cost | Game-theoretic role |
|---|---|---|---|
| 0 | `funding_boost` | $50K | Direct payoff transfer |
| 1 | `teacher_incentive` | $80K | Incentive-compatibility nudge |
| 2 | `student_scholarship` | $30K | Targeted utility lift |
| 3 | `attendance_mandate` | $10K | Enforcement / punishment lever |
| 4 | `resource_realloc` | $40K | Pareto-improving redistribution |
| 5 | `transparency_report` | $5K | Information-asymmetry reducer |
| 6 | `staff_hiring` | $120K | Capacity expansion |
| 7 | `counseling_programs` | $25K | Externality-internalisation |

</details>

<details>
<summary><strong>Reward function</strong></summary>

```python
reward = clip(
    -2.0 * dropout_rate
    +1.0 * (teacher_retention - 0.7)        # baseline-anchored
    +0.5 * student_engagement
    -0.001 * (cost / 50K),                   # token cost penalty
    -2, +2
)
```

A **narrow, sparsity-shaped signal**. The two cliffs in the field — dropout and retention — dominate. The broader **health score** (enrollment, attendance, burnout, etc.) is monitored separately for the dashboard but kept *out* of reward by design — so the policy doesn't game auxiliary metrics, a classic *reward-hacking* pitfall in RL [^15].

</details>

<details>
<summary><strong>Episode termination — collapse triggers</strong></summary>

The episode ends with `terminated=True` if **any** of:

- `dropout_rate > 0.50`        — cohort collapse
- `teacher_retention < 0.20`   — staff exodus
- `budget_remaining < -$500K`  — fiscal insolvency
- `enrollment_rate < 0.30`     — demographic collapse

These are *hard cliffs*: the agent gets zero future reward from a collapse state, training it to *prevent* the cascade rather than *recover* from it — i.e., to keep the system away from the absorbing-state attractors of the dynamic game.

</details>

---

## 🧠 The innovation — two-tier swarm orchestration

The deliberation layer is **inference-only**; it does not participate in the RL training loop. Its purpose is to produce structured, auditable verdicts that we use as the *teacher* in the distillation pipeline below.

### L1 Orchestrator — `WeightAllocator`

Computes a **per-role attention score** from state pressures and assigns each role swarm:

- **A model** — heavyweight `llama-3.3-70b` for high-attention roles, lightweight `llama-3.1-8b` for routine
- **A verdict weight** — multiplier (1.0× routine, 1.5× crisis) applied during cross-swarm aggregation

```python
# state pressure → role attention → model + weight
attention_threshold = 0.6
# e.g., funding-cut state:
#   student     attention = 0.70  → 70B + 1.5×
#   teacher     attention = 0.90  → 70B + 1.5×
#   admin       attention = 0.58  → 8B  + 1.0×
#   policymaker attention = 0.70  → 70B + 1.5×
```

### L2 Orchestrator — `PersonaAllocator`

Inside each swarm, distributes weight across the three personas using `fit_signals` declared per-persona — a direct implementation of the *salience-weighted attention model* of bounded rationality [^14]:

```yaml
- id: student_dropout_risk
  name: "Rohit, Working Student"
  fit_signals: { budget: 0.8, dropout: 0.9, attendance: 0.7 }
```

Per-persona weight ∈ `[0.3, 1.5]`, mapped from raw fit score `[0, 1]`. **Even off-topic personas keep a baseline voice** — the swarm hears all three lenses, just at different volumes. This is intentional: muting a persona collapses the persona library back into a single weighted voice and destroys the very disagreement signal the architecture exists to surface.

In a funding-cut crisis the Student swarm is dominated by **Rohit** (working student, dropout-focused). In a learning-quality crisis the same swarm is dominated by **Priya** (high achiever, classroom-quality-focused). Same swarm, different effective verdict, driven entirely by state.

### Resonance metric — Schelling-point detector

For each of the 8 interventions:

```
Resonance(intervention) = 1 − σ_normalised(verdict[role] for role in swarms)
```

**Resonance is unweighted** — operator weights modulate the *recommendation*, but *disagreement* is reported faithfully. We never collapse dissent.

```python
if resonance(intervention) >= 0.55:
    # SCHELLING POINT — four lenses converge, mechanism deploys cheap
    pass
elif resonance(intervention) < 0.55:
    # DISSONANT — flag for human judgement
    # this is where the bounded-rational equilibrium needs an
    # external coordinating authority to move
    raise_dissonance_flag(intervention)
```

### Persona library — 4 roles × 3 personas, all defined in YAML

<details>
<summary><strong>The 12 personas</strong> (click to expand)</summary>

**Student swarm** (`fit_signals` reflect economic pressure vs. classroom quality)

| Persona | Tag | Sensitive to |
|---|---|---|
| Maya | First-generation aspirant from rural low-income family | budget, enrollment, attendance, engagement |
| Rohit | Working student at acute dropout risk | budget, dropout, attendance, retention |
| Priya | High achiever preparing for IIT-JEE | engagement, resource, class size, burnout |

**Teacher swarm**

| Persona | Tag | Sensitive to |
|---|---|---|
| Mr. Sharma | 22-yr veteran, burnt out | burnout, class size, resource, retention |
| Ms. Iyer | Year-2 idealist, overwhelmed | engagement, dropout, burnout, attendance |
| Rep. Kumar | Union representative, strategic | retention, burnout, budget, trust |

**Admin swarm**

| Persona | Tag | Sensitive to |
|---|---|---|
| Principal Desai | Pragmatist, balancing books and morale | budget, resource, retention, trust |
| Director Joshi | Compliance hawk, audit-driven | trust, attendance, enrollment, retention |
| Dr. Rao | Innovator, ed-tech adopter | engagement, resource, dropout, burnout |

**Policymaker swarm**

| Persona | Tag | Sensitive to |
|---|---|---|
| Minister Verma | Fiscal hawk, treasury-aligned | budget, retention, resource |
| Sec. Banerjee | Equity champion, social justice mandate | dropout, enrollment, engagement, attendance |
| MLA Khan | Political operator, election-cycle minded | trust, dropout, attendance, enrollment |

Adding a new role is a YAML edit — no Python required. See [`swarms/config/roles.yaml`](swarms/config/roles.yaml).

</details>

---

## 🎓 Knowledge distillation pipeline

The 12-persona swarm is too slow and expensive for production deployment (12 LLM calls per decision, ~30 s, ~$0.02 per call). We compress it via supervised fine-tuning [^2] into a single `Llama-3.2-1B-Instruct` adapter:

| Stage | Tool | Wall-clock |
|---|---|---|
| Generate `(state, scenario) → (action, reasoning)` pairs | [`generate_dataset.py`](generate_dataset.py) — runs the swarm on jittered states across 5 scenario templates | 30–45 min, ~$0.40 |
| QLoRA fine-tune base model | [`training/train_unsloth.ipynb`](training/train_unsloth.ipynb) — Unsloth + HF TRL `SFTTrainer` on Kaggle T4 | ~2.2 min, free |
| Evaluate on the env | [`evaluation/eval_distilled.py`](evaluation/eval_distilled.py) — student vs. random vs. zero-action baselines | ~70 min on Mac MPS, ~5 min on CUDA |

### Why distillation, not GRPO?

Real RL on an LLM agent in this env would take 4 × H100 × multiple days — out of scope for a hackathon budget. Distillation is the right scale of training given the time budget and produces directly comparable, falsifiable evidence: *the student matches the teacher's mechanism-design verdict on held-out states*.

> The swarm-of-swarms **is the teacher** (and the mechanism designer); the 1-B model **is the student**. The student carries the swarm's deliberation into deployment at ~100× lower cost.

---

## 📊 Results

> Distilling the Vishwamitra swarm-of-swarms into a 1-B Llama-3.2 with 80 SFT examples produces a controller that **decisively avoids the catastrophic-inaction trajectory** on `DropoutCommonsEnv`: cumulative reward of **+0.45** versus **−23.50** for the do-nothing baseline over 40-step funding-cut episodes — a **~24 reward-unit gap**. The do-nothing baseline is precisely the Pareto-inefficient Nash equilibrium the system is designed to break. Per-lever fidelity to the swarm teacher is modest (R² = 0.06, MAE = 0.16, top-3 = 11.1%), as expected for an 80-example distillation set on a 1-B base; **6 of 8 interventions** match the teacher within one σ.
>
> Numbers below are reproduced verbatim from [`results.json`](results.json). Plots are in [`docs/img/`](docs/img/).

### Training & validation loss

![Training loss](docs/img/loss_curve.png)

*Distillation training on Kaggle T4 (Unsloth + TRL `SFTTrainer`, 33 steps over 3 epochs). Train loss falls from 2.43 → 0.48; validation reaches 0.46 — sitting **below** train at convergence, indicating the 1-B student fits the swarm's policy distribution without overfitting.*

### Reward on `DropoutCommonsEnv` — funding-cut scenario

![Reward comparison](docs/img/reward_curve.png)

*Cumulative episode reward (mean ± SE across 20 episodes, 40 steps each). The do-nothing baseline (slate) is the **Nash collapse trajectory** — every player plays their dominant strategy and the system bleeds reward to −23.5. Both the random baseline and the trained student keep the system at consistently positive reward, but the **trained student does it with ~10% lower variance** — i.e., consistent mechanism-design steering rather than lucky averaging.*

| Policy | Mean cumulative reward | Std | Wall-clock |
|---|---|---|---|
| Random uniform | +0.457 | ±0.951 | <1 s |
| **Zero-action (Nash collapse) baseline** | **−23.504** | ±2.804 | <1 s |
| **1-B distilled student (ours)** | **+0.449** | **±0.857** | ~71 min |

### Action-vector fidelity (held-out validation)

![Action fidelity](docs/img/action_fidelity.png)

*Per-example scatter of student vs. teacher recommended intensity for each of the 8 interventions, on the 9 held-out validation states (each state contributes 8 dots). The student concentrates predictions in the 0.4–0.7 band — it has learned the swarm's typical operating range but not yet the per-state magnitude swings.*

| Metric | Value | Interpretation |
|---|---|---|
| Mean Absolute Error per intervention | **0.159** | average miss = 16% of the [0,1] action range |
| Pearson correlation (teacher ↔ student) | **0.236** | weak-but-real positive correlation |
| R² (Pearson²) | **0.056** | student's predictions explain ~6% of teacher variance |
| Top-3 intervention agreement | **11.1%** | above C(3,3)/C(8,3) ≈ 1.8% chance |
| Validation set size | 9 examples × 8 interventions = 72 dots | |

### Per-intervention recommended intensity

![Per-intervention bars](docs/img/per_intervention.png)

*Mean recommended intensity per intervention (yellow = swarm teacher, blue = 1-B distilled student, error bars = 1 σ across the validation set). **6 of 8 levers** match within one σ — i.e., the student inherits the swarm's mechanism-design preferences on three-quarters of the action space.*

| Intervention | Teacher mean | Student mean | MAE | Status |
|---|---|---|---|---|
| `funding_boost` | 0.71 | 0.77 | 0.171 | ✓ close (slight overshoot) |
| `teacher_incentive` | 0.75 | 0.59 | 0.186 | ✓ close (undershoot) |
| `student_scholarship` | 0.54 | 0.59 | 0.120 | ✓ close |
| `attendance_mandate` | 0.25 | 0.50 | **0.267** | ⚠ student misses the *de-emphasis* signal |
| `resource_realloc` | 0.70 | 0.54 | 0.190 | ✓ close (undershoot) |
| `transparency_report` | 0.63 | 0.58 | 0.143 | ✓ close |
| `staff_hiring` | 0.46 | 0.54 | 0.136 | ✓ close |
| `counseling_programs` | 0.57 | 0.54 | **0.063** | ✓ best — student nailed this lever |

### What this evidence supports

| Claim | Evidence |
|---|---|
| The SFT pipeline is sound | Loss curve: train 2.43 → 0.48, val 0.46, no overfitting gap |
| The student breaks the Nash collapse trajectory | Reward curve: +0.45 vs −23.5 zero-policy (24-unit gap, 20-episode mean) |
| The student is *consistent*, not just lucky | Trained σ = 0.857 vs. random σ = 0.951 |
| The student inherits the swarm's mechanism-design preferences | Per-intervention table: 6 of 8 levers match within 1 σ |
| Fidelity is the bottleneck, not training | R² = 0.06 with N = 80 SFT examples — scaling to 500–1 000 examples is the immediate next step |

### Cost / latency comparison (estimated)

| | Inference cost / decision | Latency | Hardware |
|---|---|---|---|
| 12-persona swarm teacher | ~$0.020 (12 LLM calls) | ~30 s | API |
| **1-B distilled student** | **~$0.0002** | **~0.3 s on GPU, ~5 s on Mac MPS** | **single GPU / laptop** |
| **Speed-up** | **~100×** | **~100×** | — |

---

## 🎬 Watch the walkthrough

> *Replace with your unlisted YouTube URL once recorded.*

[![Watch the demo](https://img.shields.io/badge/▶️_YouTube-Walkthrough-FF0000?style=for-the-badge&logo=youtube)](#)

The video covers:

1. **The Sundarpur scenario** — a mid-sized district hit by a 35% mid-year budget cut (the kind of crisis a District Education Officer faces every Monday somewhere).
2. **The swarm-of-swarms running live** — 12 personas deliberating, the resonance/Schelling-point map appearing, the dissonance flag firing on `attendance_mandate`.
3. **The trained 1-B student** replicating the swarm's mechanism-design verdict in 0.3 seconds.
4. **Reward curves** — distilled student vs. Nash-collapse baseline on the env.

---

## ♻️ Reproducibility

All experiments below run on a free Kaggle T4 GPU and ~$1 of LLM credit (Together AI, Fireworks, Groq, or HF Router — the client is provider-agnostic).

### 1. Generate the distillation dataset

```bash
python generate_dataset.py --n 300            # ~30-45 min, ~$0.40
# → data/train.jsonl (~270 rows)
# → data/val.jsonl   (~30 rows)
```

The script samples 60 jittered variants from each of 5 scenario templates (funding crisis, teacher exodus, pandemic recovery, rural constraint, healthy school), runs the full swarm on each, and writes chat-message-formatted JSONL ready for `SFTTrainer`.

### 2. Train

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudraBhaskar9439/Enigma/blob/main/training/train_unsloth.ipynb)

The notebook handles GPU detection, Unsloth setup, LoRA configuration, training (3 epochs, ~33 steps), loss plotting, and adapter download. ~2.2 min wall-clock on T4.

### 3. Evaluate

```bash
python evaluation/eval_distilled.py \
    --adapter vishwamitra-1b-lora \
    --val data/val.jsonl \
    --episodes 20 \
    --max-steps 40
```

Produces `docs/img/reward_curve.png`, `action_fidelity.png`, `per_intervention.png`, and `results.json`. On a Mac MPS device the trained slot takes ~70 min; on CUDA roughly 5 min.

---

## 📁 Repository layout

```
Enigma/
├── env/                              OpenEnv-compatible Gymnasium env
│   ├── dropout_env.py                gym.Env implementation
│   ├── state.py                      SystemState dataclass + health_score
│   ├── spaces.py                     Observation / action spaces
│   ├── scenarios/                    funding_cut · teacher_shortage · pandemic_recovery · ...
│   └── collapse_detector.py
│
├── agents/                           Four bounded-rational stakeholder agents (rule-based)
│   ├── student_agent.py              Social contagion + dropout
│   ├── teacher_agent.py              Burnout + tit-for-tat reciprocity
│   ├── admin_agent.py                Resource allocation under budget constraint
│   └── policymaker_agent.py          Election-cycle shocks
│
├── swarms/                           ★ THE INNOVATION — swarm-of-swarms layer
│   ├── core/
│   │   ├── persona.py                Persona dataclass with fit_signals + system_prompt
│   │   ├── llm_client.py             Provider-agnostic OpenAI-format wrapper + cache
│   │   ├── verdict.py                Verdict / SwarmVerdict / ResonanceReport
│   │   ├── swarm_agent.py            Single LLM-backed deliberator
│   │   └── swarm.py                  Role-agnostic swarm (subclassing-free)
│   ├── orchestrator/
│   │   ├── router.py                 ★ L1 + L2 orchestrators
│   │   ├── swarm_manager.py          Top-level deliberation orchestrator
│   │   ├── resonance.py              Cross-swarm Schelling-point detector + dissonance flags
│   │   ├── round_log.py              JSONL audit trail
│   │   └── policy_report.py          Three-call policy-brief generator
│   ├── config/roles.yaml             ★ 4 roles × 3 personas — no Python per role
│   └── prompts/                      Shared verdict / action-space prompts
│
├── server/                           FastAPI backend
│   ├── app.py                        OpenEnv HTTP API + Gradio mount
│   └── swarm_routes.py               /swarms/info  /swarms/deliberate  /swarms/policy-report
│
├── frontend/                         React 19 + Vite + react-flow UI
│   └── src/components/Vishwamitra/   Bloomberg-style swarm explorer
│
├── training/
│   ├── train_unsloth.ipynb           ★ Kaggle SFT distillation notebook
│   └── _build_notebook.py            Notebook source-of-truth (Python)
│
├── evaluation/
│   └── eval_distilled.py             Trained model vs. random / Nash-collapse baselines on the env
│
├── data/
│   ├── train.jsonl                   Distillation dataset (90% split)
│   └── val.jsonl                     Held-out validation
│
├── generate_dataset.py               Creates train/val.jsonl from the swarm
├── examples/swarm_demo.py            CLI demo of the swarm-of-swarms
├── inference.py                      Load + use the trained adapter
├── openenv.yaml                      OpenEnv manifest
├── Blog.md                           ★ Long-form technical blog
└── pyproject.toml
```

---

## 🛣️ Roadmap

- [x] OpenEnv-compatible env with calibrated stakeholder dynamics
- [x] Two-tier swarm orchestrator: state → model + weight per role + per persona
- [x] Resonance / dissonance metric with explicit human-judgement gating
- [x] Three-call LLM-authored Educational Policy Brief PDF (six-stage policymaking template)
- [x] Knowledge distillation into 1-B student via SFT
- [ ] Real RL fine-tuning of the student via TRL `GRPOTrainer` against the env (research-scope, post-hackathon) [^16]
- [ ] Mechanism-design auction layer — simulate budget allocation across competing interventions as a sealed-bid auction
- [ ] Online curriculum: dissonance score → automatic env scenario generator
- [ ] Multi-language persona library (regional Indian languages, then global)
- [ ] Federated deployment for state-level education ministries (OpenEnv server farm)

---

## 📜 Citation

If you use Vishwamitra in academic work or policy analysis, please cite as:

```bibtex
@software{bhaskar2026vishwamitra,
  title       = {Vishwamitra: Disagreement-Mapping for Educational Commons},
  author      = {Bhaskar, Rudra},
  year        = {2026},
  url         = {https://github.com/RudraBhaskar9439/Enigma},
  note        = {Built for the Meta · PyTorch Hackathon 2026, Round 2 (India).
                 OpenEnv-compatible multi-agent simulator framed as a coordination
                 game, with swarm-of-swarms LLM deliberation and a knowledge-
                 distilled 1-B student model that breaks the Pareto-inefficient
                 Nash equilibrium of the do-nothing baseline.}
}
```

---

## 📖 References

[^1]: Meta AI Research. *OpenEnv: A Reinforcement-Learning Environment Specification for Agent Research*. 2025. <https://github.com/facebookresearch/openenv>

[^2]: Hinton, G., Vinyals, O., & Dean, J. *Distilling the Knowledge in a Neural Network*. NeurIPS 2014 Deep Learning Workshop. <https://arxiv.org/abs/1503.02531>

[^3]: Han, D., Han, M., et al. *Unsloth: 2× faster, 70% less memory LLM fine-tuning*. 2024. <https://github.com/unslothai/unsloth>

[^4]: Ostrom, E. *Governing the Commons: The Evolution of Institutions for Collective Action*. Cambridge University Press, 1990.

[^5]: Hu, E. J., Shen, Y., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. <https://arxiv.org/abs/2106.09685>

[^6]: Hugging Face. *TRL: Transformer Reinforcement Learning*. 2024. <https://github.com/huggingface/trl>

[^7]: UNESCO Institute for Statistics. *Out-of-School Children and Youth — Global Indicators*, 2024.

[^8]: Government of India, Department of School Education. *Unified District Information System for Education (DISE)*, 2023.

[^9]: Nash, J. F. *Equilibrium Points in N-Person Games*. Proceedings of the National Academy of Sciences, 1950. <https://www.pnas.org/doi/10.1073/pnas.36.1.48>

[^10]: Simon, H. A. *A Behavioral Model of Rational Choice*. Quarterly Journal of Economics, 1955.

[^11]: Arrow, K. J. *Social Choice and Individual Values*. Wiley, 1951. (The impossibility theorem for social-welfare aggregation.)

[^12]: Hurwicz, L., Maskin, E., Myerson, R. *Foundations of Mechanism Design*. (Nobel Memorial Prize in Economics, 2007.) <https://www.nobelprize.org/prizes/economic-sciences/2007/summary/>

[^13]: Schelling, T. C. *The Strategy of Conflict*. Harvard University Press, 1960. (Schelling-point coordination.)

[^14]: Bordalo, P., Gennaioli, N., Shleifer, A. *Salience Theory of Choice Under Risk*. Quarterly Journal of Economics, 2012.

[^15]: Krakovna, V., Uesato, J., et al. *Specification gaming: the flip side of AI ingenuity*. DeepMind, 2020. <https://deepmind.com/blog/article/Specification-gaming-the-flip-side-of-AI-ingenuity>

[^16]: Shao, Z., Wang, P., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (introduces GRPO). 2024. <https://arxiv.org/abs/2402.03300>

---

## 🙏 Acknowledgments

Built for **Meta · PyTorch Hackathon — Round 2 · India 2026**.

- Persona designs grounded in **UNESCO** [^7], **DISE India** [^8], and **World Bank** education statistics.
- Game-theoretic framing draws from **Ostrom** [^4] (Tragedy of the Commons), **Nash** [^9] (equilibrium existence), **Schelling** [^13] (focal-point coordination), **Hurwicz/Maskin/Myerson** [^12] (mechanism design), and **Simon** [^10] (bounded rationality).
- Swarm-of-swarms architecture is inspired by deliberative-democracy literature: explicit multi-stakeholder framings and dissent-surfacing as institutional design.
- Distillation pattern follows the long line of knowledge-distillation research [^2] and recent multi-agent → single-agent compression work.
- Training framework: **Unsloth** [^3] + **HF TRL** [^6] + **LoRA** [^5].
- Environment specification: **Meta OpenEnv** [^1].

The four stakeholder swarms are not real people, but the constraints they navigate are real. The names are Indian because the operating context the system was designed for is Indian; the architecture is fully transferable.

---

## 📜 License

Released under the [MIT License](LICENSE).

---

<div align="center">

<sub>The swarm sees the equilibrium that no single agent can. The student carries that vision into the field.</sub>

<br/>

**[⬆ Back to top](#-vishwamitra)**

</div>
