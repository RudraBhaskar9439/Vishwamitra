"""
generate_dataset.py — Build a synthetic SFT dataset for distilling the
Vishwamitra swarm's policy into a small LLM (Llama-3.2-1B / Qwen2.5-0.5B).

Each example: (state vector + scenario brief) → (action_vector + reasoning),
formatted as a chat-message sequence ready for HF TRL's SFTTrainer.

Usage:
    python generate_dataset.py --n 300 --out data/

Defaults give:
  - 5 scenario templates × 60 jittered variants = 300 examples
  - 90/10 stratified train/val split
  - data/train.jsonl + data/val.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarms import SwarmManager  # noqa: E402
from swarms.core.verdict import ACTION_NAMES  # noqa: E402
from swarms.orchestrator.resonance import compute_resonance  # noqa: E402


# Hard-pin model used for the dataset: stays inside Groq 8B's daily quota
# even after a heavy day of testing. Override via REPORT_MODEL_NAME env if
# you want to use the 70B once its TPD resets.
DATASET_MODEL = "llama-3.1-8b-instant"


async def custom_deliberate(
    manager: SwarmManager,
    state: dict,
    scenario: str,
    *,
    concurrency: int = 4,
    model_override: str | None = None,
):
    """Custom deliberation that controls how many swarms fire in parallel.

      concurrency=4  → all 4 swarms in parallel (12 in-flight calls).
                       Use on Together / Fireworks / fresh Groq quota.
      concurrency=2  → swarms in pairs (6 in-flight). Use on throttled Groq.
      concurrency=1  → one swarm at a time (3 in-flight). Last resort.

    `model_override` (if given) forces every swarm to use that model
    (skipping the L1 router's choice). Useful when one tier is rate-limited.
    """
    plan = manager.router.allocate(state=state)
    l2_per_swarm = {
        s.role: manager.persona_router.allocate(
            personas=[a.persona for a in s.agents], state=state,
        )
        for s in manager.swarms
    }

    async def run_one(s):
        return await s.deliberate(
            state_snapshot=state,
            scenario=scenario,
            action_space_doc=manager._action_space_doc,
            verdict_instructions=manager._verdict_instructions,
            model=model_override or plan.model_for(s.role),
            persona_weights={d.persona_id: d.weight for d in l2_per_swarm[s.role]},
            persona_fits=[d.to_dict() for d in l2_per_swarm[s.role]],
        )

    swarm_verdicts: list = []
    swarms = manager.swarms
    step = max(1, concurrency)
    for i in range(0, len(swarms), step):
        batch = swarms[i : i + step]
        results = await asyncio.gather(*(run_one(s) for s in batch))
        swarm_verdicts.extend(results)
        if i + step < len(swarms):
            await asyncio.sleep(0.3)

    swarm_weights = {role: plan.weight_for(role) for role in plan.decisions}
    report = compute_resonance(
        swarm_verdicts=swarm_verdicts,
        dissonance_threshold=manager.dissonance_threshold,
        scenario=scenario,
        state_snapshot=state,
        swarm_weights=swarm_weights,
    )
    report.orchestrator_plan = plan.to_dict()
    return report


# ===========================================================================
# 1. Default state ranges + per-scenario overrides
# ===========================================================================

DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "enrollment_rate":     (0.70, 0.85),
    "attendance_rate":     (0.65, 0.80),
    "dropout_rate":        (0.10, 0.20),
    "teacher_retention":   (0.70, 0.85),
    "teacher_burnout":     (0.40, 0.60),
    "student_engagement":  (0.55, 0.75),
    "resource_allocation": (0.50, 0.70),
    "avg_class_size":      (32, 42),
    "budget_remaining":    (700_000, 1_000_000),
    "trust_score":         (0.60, 0.80),
}

INT_FIELDS = {"avg_class_size"}
ROUND_TO_THOUSAND = {"budget_remaining"}

SCENARIO_TEMPLATES: dict[str, dict[str, Any]] = {

    "funding_crisis": {
        "overrides": {
            "budget_remaining":    (200_000, 600_000),
            "dropout_rate":        (0.20, 0.35),
            "teacher_burnout":     (0.55, 0.85),
            "teacher_retention":   (0.55, 0.75),
            "resource_allocation": (0.30, 0.50),
            "student_engagement":  (0.40, 0.60),
            "enrollment_rate":     (0.55, 0.75),
            "attendance_rate":     (0.50, 0.70),
            "avg_class_size":      (42, 58),
            "trust_score":         (0.40, 0.65),
        },
        "briefs": [
            "State legislature passed a mid-year 35% education budget cut to plug a fiscal hole. "
            "Class sizes are growing, teacher resignations are rising, and dropout signals are "
            "spiking in 9th and 10th grade. Decide intervention intensities for the next quarter.",
            "A sudden 30% reduction in district education funding has hit the school 8 weeks "
            "into the academic year. Reserves are draining, teachers are stretched thin, and the "
            "early-warning dashboard is flagging multiple schools yellow.",
            "Funding crisis: discretionary funds redirected to other welfare schemes. Schools "
            "have roughly three months of operating budget left before payroll constraints force "
            "layoffs.",
            "The state cut education spending unexpectedly. The DEO has 90 days until a board "
            "review and must choose where to allocate the remaining $400K across eight "
            "intervention levers.",
            "Budget shock — operating funds halved at mid-year. Teacher exits are accelerating, "
            "class sizes are ballooning, and dropout warnings are concentrated in 9th and 10th "
            "grade.",
        ],
    },

    "teacher_exodus": {
        "overrides": {
            "teacher_retention": (0.30, 0.55),
            "teacher_burnout":   (0.70, 0.90),
            "avg_class_size":    (50, 60),
            "budget_remaining":  (700_000, 1_100_000),
            "dropout_rate":      (0.10, 0.20),
            "enrollment_rate":   (0.78, 0.90),
        },
        "briefs": [
            "Thirty percent of the teaching staff have given notice in the last 60 days. Exit "
            "interviews cite burnout, stagnant salaries, and lack of administrative backup. The "
            "academic year is half over; another wave will collapse the schedule.",
            "Mass resignation event in progress: 15 of 50 teachers gave notice this term. Class "
            "sizes have already grown from 35 to 52 students. The remaining staff are showing "
            "acute burnout.",
            "Retention is collapsing — 40% drop in teacher headcount this year. Budget is intact "
            "but the system is bleeding human capital faster than it can hire replacements.",
            "Teacher attrition crisis: senior staff retiring early, mid-career teachers leaving "
            "for the private sector. Burnout index is at 0.85 across the workforce.",
            "Workforce hemorrhage. Three principals requested transfers; two department heads "
            "quit. The remaining teachers are covering 1.5x their normal load.",
        ],
    },

    "pandemic_recovery": {
        "overrides": {
            "student_engagement": (0.30, 0.50),
            "trust_score":        (0.35, 0.55),
            "enrollment_rate":    (0.85, 0.95),
            "attendance_rate":    (0.65, 0.85),
            "dropout_rate":       (0.15, 0.25),
            "teacher_burnout":    (0.45, 0.70),
            "budget_remaining":   (600_000, 900_000),
        },
        "briefs": [
            "School reopened 18 months ago after extended pandemic closure. Standardized "
            "assessments show 1.4 grade levels of learning loss across cohorts. Attendance has "
            "recovered but students are mentally checked out.",
            "Post-COVID return: students physically present but disengaged. Mental-health "
            "referrals are up 220% over pre-pandemic. Teachers report classrooms feel "
            "'half-asleep'.",
            "Two years post-reopening, engagement has not recovered. Trust in the school has "
            "eroded — parents are withholding extracurricular fees, and anonymous complaints "
            "are rising.",
            "Pandemic recovery phase: enrollment is back but the cohort is academically and "
            "emotionally fragile. Dropout signals concentrate in students who lost a parent "
            "during the pandemic.",
            "Learning loss plus a mental-health crisis. Engagement sits at 0.40 versus the "
            "pre-pandemic 0.75. Some recovery funds are available but no playbook for emotional "
            "re-engagement is proven.",
        ],
    },

    "rural_constraint": {
        "overrides": {
            "resource_allocation": (0.20, 0.40),
            "budget_remaining":    (100_000, 300_000),
            "dropout_rate":        (0.18, 0.30),
            "trust_score":         (0.65, 0.85),
            "enrollment_rate":     (0.65, 0.80),
            "attendance_rate":     (0.55, 0.75),
            "teacher_retention":   (0.60, 0.80),
            "avg_class_size":      (38, 52),
        },
        "briefs": [
            "Rural district serving four villages, no broadband, single bus route. Sixty percent "
            "of families are below the poverty line. Government has redirected discretionary "
            "funds to urban districts for a third consecutive year.",
            "Remote agricultural region: school building unpainted in eight years, library has "
            "200 books for 800 students. Teachers commute 90 minutes each way. Community trust "
            "is high but resources are minimal.",
            "Tribal-area school: chronic under-resourcing, structural rather than acute. "
            "Operating on a shoestring; dropouts are driven by economic necessity, not academic "
            "disengagement.",
            "Rural district with two secondary schools serving 1,500 students across 30 villages. "
            "Bus reliability is the largest dropout driver. Resource allocation has been "
            "chronically below state minimum.",
            "Resource-poor remote setting. Strong community trust, weak infrastructure. "
            "Students walk up to six kilometres each way; transport budget is exhausted by "
            "November.",
        ],
    },

    "healthy_school": {
        "overrides": {
            "dropout_rate":        (0.03, 0.10),
            "teacher_retention":   (0.85, 0.95),
            "teacher_burnout":     (0.20, 0.40),
            "student_engagement":  (0.75, 0.90),
            "enrollment_rate":     (0.88, 0.96),
            "attendance_rate":     (0.85, 0.95),
            "resource_allocation": (0.70, 0.85),
            "budget_remaining":    (1_200_000, 1_800_000),
            "avg_class_size":      (25, 35),
            "trust_score":         (0.80, 0.95),
        },
        "briefs": [
            "Routine mid-year check-in. The school is running well, with no acute crises. "
            "Performance is steady, retention is strong, the budget is healthy. The board is "
            "asking what light-touch maintenance the system should prioritise.",
            "Quarterly review at a high-performing district school. All metrics are within "
            "healthy bands. The DEO wants to identify which 2-3 marginal improvements would "
            "most efficiently raise outcomes.",
            "Stable academic year, mid-term assessment. No fires to fight. The question is "
            "forward investment: where would an extra $200K of discretionary spending move the "
            "needle most?",
            "Standard monitoring cycle. Engagement, retention, and dropout indicators are all "
            "green. The budget is under-utilised by roughly 12%. Where should the surplus go?",
            "The school is on a stable trajectory after several improvement years. The DEO "
            "wants to consolidate gains and pre-empt slow-creep risks (e.g. teacher burnout "
            "starting to rise).",
        ],
    },
}


# ===========================================================================
# 2. State sampling
# ===========================================================================

def _sample_field(lo: float, hi: float, rng: random.Random, integer: bool, round_to_thousand: bool) -> float:
    mid = (lo + hi) / 2.0
    sigma = (hi - lo) * 0.30
    v = rng.gauss(mid, sigma)
    v = max(lo, min(hi, v))
    if integer:
        return int(round(v))
    if round_to_thousand:
        return float(round(v / 1000.0) * 1000.0)
    return round(v, 3)


def sample_state(template_name: str, rng: random.Random) -> dict[str, float]:
    overrides = SCENARIO_TEMPLATES[template_name]["overrides"]
    ranges = {**DEFAULT_RANGES, **overrides}
    state: dict[str, float] = {}
    for field, (lo, hi) in ranges.items():
        state[field] = _sample_field(
            lo, hi, rng,
            integer=(field in INT_FIELDS),
            round_to_thousand=(field in ROUND_TO_THOUSAND),
        )
    state["step"] = rng.randint(5, 30)
    return state


def sample_brief(template_name: str, rng: random.Random) -> str:
    return rng.choice(SCENARIO_TEMPLATES[template_name]["briefs"])


# ===========================================================================
# 3. Reasoning synthesizer (programmatic — no extra LLM call)
# ===========================================================================

def synthesize_reasoning(report_dict: dict[str, Any]) -> str:
    final = report_dict.get("final_action") or [0.0] * 8
    names = report_dict.get("action_names") or list(ACTION_NAMES)
    flags = report_dict.get("dissonance_flags") or []

    ranked = sorted(
        zip(names, final), key=lambda x: -float(x[1])
    )
    top = [n for n, v in ranked if v >= 0.5][:3]
    if not top:
        # nothing strongly recommended
        top = [ranked[0][0]] if ranked else []

    swarm_peaks: list[str] = []
    for sv in report_dict.get("swarm_verdicts", []):
        verdicts = sv.get("verdicts") or []
        if not verdicts:
            continue
        peak = max(verdicts, key=lambda v: v.get("confidence", 0.0))
        if peak.get("error") or (peak.get("confidence", 0.0) <= 0.2):
            continue
        first_name = (peak.get("persona_name") or "").split(",")[0].strip()
        if first_name:
            swarm_peaks.append(f"{first_name} ({sv.get('role','?')})")

    parts: list[str] = []
    if top:
        top_str = ", ".join(
            f"{n} ({float(v):.2f})"
            for n, v in ranked
            if n in top
        )
        parts.append(f"Recommended interventions: {top_str}.")

    if swarm_peaks:
        parts.append(
            f"Strongest voices: {'; '.join(swarm_peaks[:3])}."
        )

    if flags:
        parts.append(
            f"Dissonant on {', '.join(flags)} — operator judgement required."
        )
    else:
        parts.append("All swarms resonated; high-confidence consensus.")

    return " ".join(parts)


# ===========================================================================
# 4. Chat-message formatting
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an educational policy assistant. Given a system state and a "
    "scenario brief, recommend intervention intensities (each in [0, 1]) and "
    "provide a brief rationale. Output a single JSON object with keys "
    "`action_vector` (eight named floats) and `reasoning` (1-3 sentences)."
)


def format_input(state: dict[str, Any], scenario: str) -> str:
    state_lines = "\n".join(f"  - {k}: {state[k]}" for k in state)
    return (
        f"STATE:\n{state_lines}\n\n"
        f"SCENARIO: {scenario}\n\n"
        "Respond with a single JSON object in this exact shape:\n"
        '{"action_vector": {"funding_boost": <float>, '
        '"teacher_incentive": <float>, "student_scholarship": <float>, '
        '"attendance_mandate": <float>, "resource_realloc": <float>, '
        '"transparency_report": <float>, "staff_hiring": <float>, '
        '"counseling_programs": <float>}, '
        '"reasoning": "<one to three sentences>"}'
    )


def format_output(report_dict: dict[str, Any], reasoning: str) -> str:
    final = report_dict.get("final_action") or [0.0] * 8
    names = report_dict.get("action_names") or list(ACTION_NAMES)
    av = {n: round(float(v), 2) for n, v in zip(names, final)}
    payload = {"action_vector": av, "reasoning": reasoning}
    return json.dumps(payload, ensure_ascii=False)


def to_message_row(
    state: dict[str, Any], scenario: str, report_dict: dict[str, Any], template: str
) -> dict[str, Any]:
    reasoning = synthesize_reasoning(report_dict)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_input(state, scenario)},
            {"role": "assistant", "content": format_output(report_dict, reasoning)},
        ],
        "_template": template,
    }


# ===========================================================================
# 5. Generation loop
# ===========================================================================

def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "_checkpoint.jsonl"


def _append_checkpoint(out_dir: Path, row: dict[str, Any]) -> None:
    """Persist each successful row immediately so a Ctrl+C / quota-hit
    can never destroy work-in-progress."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with _checkpoint_path(out_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_checkpoint(out_dir: Path) -> list[dict[str, Any]]:
    p = _checkpoint_path(out_dir)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


async def generate(
    n_per_template: int,
    sleep_between: float,
    seed: int,
    manager: SwarmManager,
    concurrency: int,
    model_override: str | None,
    min_live: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    failures = 0
    templates = list(SCENARIO_TEMPLATES.keys())
    total = n_per_template * len(templates)

    print(f"\nGenerating {total} examples ({n_per_template} per template)...")
    print(f"Templates: {', '.join(templates)}\n")

    i = 0
    start = time.time()
    for tmpl in templates:
        for _ in range(n_per_template):
            i += 1
            state = sample_state(tmpl, rng)
            brief = sample_brief(tmpl, rng)
            try:
                report = await custom_deliberate(
                    manager, state, brief,
                    concurrency=concurrency,
                    model_override=model_override,
                )
                rd = report.to_dict()
                live_verdicts = sum(
                    1
                    for sv in rd.get("swarm_verdicts", [])
                    for v in sv.get("verdicts", [])
                    if not v.get("error")
                )
                if live_verdicts < min_live:
                    failures += 1
                    print(f"  [{i:>3}/{total}] {tmpl:18s}  SKIP (only {live_verdicts}/12 live)")
                    continue
                row = to_message_row(state, brief, rd, tmpl)
                rows.append(row)
                _append_checkpoint(out_dir, row)  # crash-safe persist
            except Exception as e:
                failures += 1
                msg = str(e)
                if len(msg) > 80:
                    msg = msg[:80] + "…"
                print(f"  [{i:>3}/{total}] {tmpl:18s}  FAIL: {msg}")
                continue

            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            mins = eta / 60.0
            print(
                f"  [{i:>3}/{total}] {tmpl:18s}  ok  "
                f"({len(rows)} kept, {failures} skipped)  "
                f"·  elapsed {elapsed:.0f}s  ·  ETA {mins:.1f}m"
            )

            if sleep_between > 0:
                await asyncio.sleep(sleep_between)

    return rows, failures


# ===========================================================================
# 6. Validation + stats
# ===========================================================================

def validate(rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    if n == 0:
        raise SystemExit("ERROR: zero rows generated. Aborting.")

    print("\n" + "=" * 64)
    print("DATASET VALIDATION REPORT")
    print("=" * 64)

    # Per-template count
    counts = Counter(r.get("_template", "?") for r in rows)
    print(f"\nPer-template count (target ~equal across templates):")
    for tmpl, c in sorted(counts.items()):
        print(f"  {tmpl:22s}  {c}")

    # JSON validity
    valid = 0
    parsed_outputs: list[dict[str, Any]] = []
    for r in rows:
        try:
            out = json.loads(r["messages"][-1]["content"])
            assert "action_vector" in out and "reasoning" in out
            assert all(name in out["action_vector"] for name in ACTION_NAMES)
            assert all(0.0 <= float(out["action_vector"][n]) <= 1.0 for n in ACTION_NAMES)
            valid += 1
            parsed_outputs.append(out)
        except Exception:
            pass
    print(f"\nJSON validity: {valid}/{n} ({100*valid/n:.1f}%)")
    if valid / n < 0.95:
        print("  ⚠️  Less than 95% valid — investigate before training.")

    # Action vector distribution
    print("\nAction vector distribution:")
    print(f"  {'intervention':22s}  {'mean':>5}  {'std':>5}  {'min':>5}  {'max':>5}")
    print("  " + "─" * 50)
    for name in ACTION_NAMES:
        vals = [float(o["action_vector"][name]) for o in parsed_outputs]
        m = statistics.mean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {name:22s}  {m:5.2f}  {s:5.2f}  {min(vals):5.2f}  {max(vals):5.2f}")

    # Reasoning length
    lengths = [len(r["messages"][-1]["content"]) for r in rows]
    lengths.sort()
    p95 = lengths[int(0.95 * len(lengths))] if lengths else 0
    print(
        f"\nOutput JSON length:  "
        f"mean={statistics.mean(lengths):.0f}  "
        f"p95={p95}  "
        f"max={max(lengths)} chars"
    )
    if max(lengths) > 1500:
        print("  ⚠️  Some outputs > 1500 chars — may strain 1B model context.")

    # Diversity sniff: warn if any action has very low std
    print("\nDiversity check:")
    weak_signal = []
    for name in ACTION_NAMES:
        vals = [float(o["action_vector"][name]) for o in parsed_outputs]
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        if s < 0.05:
            weak_signal.append((name, s))
    if weak_signal:
        print("  ⚠️  Low-variance actions (model may memorise constants):")
        for name, s in weak_signal:
            print(f"     {name}: std={s:.3f}")
    else:
        print("  All actions show meaningful variance across examples ✓")

    print("\n" + "=" * 64)


# ===========================================================================
# 7. Stratified split + write JSONL
# ===========================================================================

def stratified_split(
    rows: list[dict[str, Any]],
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_tmpl: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_tmpl.setdefault(r["_template"], []).append(r)
    train, val = [], []
    for items in by_tmpl.values():
        rng.shuffle(items)
        k = max(1, int(round(len(items) * val_fraction)))
        val.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def strip_internal_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": r["messages"]} for r in rows]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===========================================================================
# 8. CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Vishwamitra SFT dataset.")
    parser.add_argument("--n", type=int, default=300,
                        help="Total examples (will be split equally across "
                             f"{len(SCENARIO_TEMPLATES)} templates).")
    parser.add_argument("--out", type=Path, default=Path("data"),
                        help="Output directory (default: ./data).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between examples (helps "
                             "avoid TPM throttling).")
    parser.add_argument("--concurrency", type=int, default=4, choices=[1, 2, 4],
                        help="How many swarms run in parallel per deliberation "
                             "(4 = all parallel = 12 in-flight calls; "
                             "2 = paired = 6 in-flight; "
                             "1 = serial = 3 in-flight). Use 1-2 if your "
                             "provider rate-limits aggressively.")
    parser.add_argument("--force-model", type=str, default=None,
                        help="Override the L1 router's per-swarm model "
                             "choice (e.g. 'llama-3.1-8b-instant'). "
                             "Useful when one tier's daily quota is exhausted.")
    parser.add_argument("--min-live", type=int, default=4,
                        help="Drop examples where fewer than this many of 12 "
                             "personas returned a valid verdict.")
    args = parser.parse_args()

    n_templates = len(SCENARIO_TEMPLATES)
    n_per = args.n // n_templates
    actual = n_per * n_templates
    if actual != args.n:
        print(f"NOTE: --n={args.n} is not divisible by {n_templates}; "
              f"will generate {actual} examples ({n_per} per template).")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out.resolve()}")

    manager = SwarmManager(log=False)
    print(
        f"Provider: {manager.client.config.provider_name}  ·  "
        f"Model: {manager.client.config.model}"
    )
    print(
        f"Roles: {', '.join(manager.roles)}  "
        f"({sum(len(s.agents) for s in manager.swarms)} personas total)"
    )

    print(
        f"Concurrency: {args.concurrency} swarm(s) parallel  "
        f"({args.concurrency * 3} in-flight LLM calls)"
    )
    if args.force_model:
        print(f"Forced model: {args.force_model}")

    rng = random.Random(args.seed)
    try:
        rows, failures = asyncio.run(
            generate(
                n_per, args.sleep, args.seed, manager,
                concurrency=args.concurrency,
                model_override=args.force_model,
                min_live=args.min_live,
                out_dir=args.out,
            )
        )
    except KeyboardInterrupt:
        print("\n\nINTERRUPTED — recovering checkpoint…")
        rows = _load_checkpoint(args.out)
        failures = 0
        print(f"Recovered {len(rows)} rows from {_checkpoint_path(args.out)}")

    print("\nGeneration complete.")
    print(f"  successful: {len(rows)}")
    print(f"  skipped:    {failures}")

    if not rows:
        raise SystemExit("ERROR: no rows generated; nothing to write.")

    validate(rows)

    train, val = stratified_split(rows, args.val_fraction, rng)
    train = strip_internal_keys(train)
    val = strip_internal_keys(val)

    train_path = args.out / "train.jsonl"
    val_path = args.out / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print(f"\nWrote {len(train)} rows → {train_path}")
    print(f"Wrote {len(val)} rows → {val_path}")
    print("\n✔ Dataset ready for SFT.")
    print(f"  Next: upload {train_path} and {val_path} to your Colab notebook.")


if __name__ == "__main__":
    main()
