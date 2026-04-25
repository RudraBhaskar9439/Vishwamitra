"""
End-to-end demo of the Vishwamitra swarm layer.

Runs all 4 role swarms (12 personas total) on a sample funding-cut
scenario and pretty-prints the ResonanceReport: per-role verdicts,
final action vector, and which interventions the swarms disagreed on.

Usage:
    python -m examples.swarm_demo
    # or:
    python examples/swarm_demo.py

Requires one of: GROQ_API_KEY / TOGETHER_API_KEY / FIREWORKS_API_KEY /
HF_TOKEN / OPENAI_API_KEY in the environment (or in spaces/.env).
"""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarms import SwarmManager
from swarms.core.verdict import ACTION_NAMES


SAMPLE_STATE = {
    # Mid-crisis funding-cut scenario state
    "enrollment_rate": 0.62,
    "attendance_rate": 0.55,
    "dropout_rate": 0.28,
    "teacher_retention": 0.71,
    "budget_utilization": 0.92,
    "avg_class_size": 48.0,
    "teacher_workload": 0.85,
    "resource_allocation": 0.40,
    "student_engagement": 0.50,
    "teacher_burnout": 0.72,
    "policy_compliance": 0.65,
    "budget_remaining": 420_000.0,
    "step": 12,
    "trust_score": 0.55,
    "data_integrity": 0.85,
}

SCENARIO = (
    "Funding cut: state slashed the education budget by 35% mid-year. "
    "Class sizes ballooned, two teachers resigned last month, dropout signals "
    "rising in 9th and 10th grade. Decide intervention intensities for the "
    "next quarter."
)


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(report) -> None:
    payload = report.to_dict()
    print("=" * 78)
    print(f"VISHWAMITRA — ResonanceReport")
    print(f"scenario: {payload['scenario'][:70]}...")
    print(f"timestamp: {payload['timestamp']}")
    print("=" * 78)

    # Per-swarm verdicts
    for sv in payload["swarm_verdicts"]:
        print(f"\n── {sv['role'].upper()} swarm "
              f"(mean confidence: {sv['mean_confidence']:.2f}) ──")
        for v in sv["verdicts"]:
            tag = "·" if v["error"] is None else "×"
            print(f"  {tag} {v['persona_name']:30s}  conf={v['confidence']:.2f}")
            reasoning = (v["reasoning"] or "").replace("\n", " ")
            if reasoning:
                print(f"      “{reasoning[:160]}{'…' if len(reasoning) > 160 else ''}”")

    # Final action + resonance map
    print("\n" + "─" * 78)
    print(f"{'INTERVENTION':<22} {'FINAL':>7}  {'RESONANCE':<24}  STATUS")
    print("─" * 78)
    for i, name in enumerate(ACTION_NAMES):
        f = payload["final_action"][i]
        r = payload["resonance_per_intervention"][i]
        flag = "🔴 DISSONANT" if name in payload["dissonance_flags"] else "🟢 resonant"
        print(f"{name:<22} {f:>7.3f}  {_bar(r, 20)} {r:.2f}  {flag}")
    print("─" * 78)

    if payload["dissonance_flags"]:
        print(f"\nDissonance flags ({len(payload['dissonance_flags'])}): "
              f"{', '.join(payload['dissonance_flags'])}")
        print("→ These are the interventions where stakeholder lenses genuinely")
        print("  disagree. Treat as decision points needing human judgment.")
    else:
        print("\nAll swarms resonated. High-confidence recommendation.")


async def main() -> None:
    manager = SwarmManager()
    print(f"Configured: {json.dumps(manager.describe(), indent=2)}\n")
    print(f"Deliberating on scenario...\n")

    report = await manager.deliberate(SAMPLE_STATE, scenario=SCENARIO)
    print_report(report)

    log_file = manager.logger.log_path if manager.logger else None
    if log_file:
        print(f"\n📝 Round log written to: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())
