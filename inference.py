"""
Vishwamitra OpenEnv Round 1 — Inference Script
==============================================

Runs 3 tasks (easy / medium / hard) over the DropoutCommonsEnv using an
LLM policy via the OpenAI-compatible client.

Required environment variables (injected by the hackathon's LiteLLM proxy):
    API_BASE_URL - OpenAI-compatible base URL of the LiteLLM proxy
    API_KEY      - participant API key
    MODEL_NAME   - approved model identifier

Optional:
    LOCAL_IMAGE_NAME - only used by from_docker_image() flows

Local dev fallback: if API_KEY is not set, the script will read HF_TOKEN
from the environment / .env so the same script works against Groq /
Fireworks / etc. without renaming variables.

Logs follow the strict [START] / [STEP] / [END] stdout format.
Designed to fit 2 vCPU / 8 GB RAM, completing in well under 20 minutes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Auto-load .env if present (no-op when python-dotenv isn't installed).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# OpenEnv-required environment variables.
#
# The hackathon's LiteLLM proxy injects API_BASE_URL + API_KEY (+ MODEL_NAME)
# into the evaluation container. Phase 2 verifies that real API calls flow
# through that proxy, so we MUST read API_KEY (not just HF_TOKEN).
#
# Local development can still use a .env with HF_TOKEN as a fallback so the
# script works against Groq / Fireworks / etc. without renaming variables.
#
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME, never for the key.
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # legacy fallback for local dev

# Optional — only relevant when using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

from env.dropout_env import DropoutCommonsEnv
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from env.scenarios.pandemic_recovery import PandemicRecoveryScenario


# ---------------------------------------------------------------------------
# Task definitions — easy / medium / hard graders
# ---------------------------------------------------------------------------

ACTION_LABELS = [
    "funding_boost", "teacher_incentive", "student_scholarship",
    "attendance_mandate", "resource_realloc", "transparency_report",
    "staff_hiring", "counseling_programs",
]

OBS_LABELS = [
    "enrollment_rate", "attendance_rate", "dropout_rate", "teacher_retention",
    "budget_utilization", "class_size_norm", "teacher_workload",
    "resource_allocation", "student_engagement", "teacher_burnout",
    "policy_compliance", "budget_remaining_norm", "step",
]


@dataclass
class Task:
    task_id: str
    difficulty: str
    scenario_factory: callable
    episode_length: int
    success_threshold: float  # avg health_score over episode
    description: str

    def grade(self, avg_health: float, collapsed: bool) -> Tuple[float, bool]:
        """Agent grader: returns (score in [0,1], passed)."""
        if collapsed:
            partial = max(0.0, avg_health) * 0.4
            return partial, False
        score = min(1.0, max(0.0, avg_health / max(self.success_threshold, 1e-6)))
        passed = avg_health >= self.success_threshold
        return score, passed


TASKS: List[Task] = [
    Task(
        task_id="task_easy_funding",
        difficulty="easy",
        scenario_factory=FundingCutScenario,
        episode_length=40,
        success_threshold=0.55,
        description="Stabilize the system through a moderate funding cut.",
    ),
    Task(
        task_id="task_medium_teacher_shortage",
        difficulty="medium",
        scenario_factory=TeacherShortageScenario,
        episode_length=60,
        success_threshold=0.50,
        description="Maintain teacher retention during a staffing crisis.",
    ),
    Task(
        task_id="task_hard_pandemic",
        difficulty="hard",
        scenario_factory=PandemicRecoveryScenario,
        episode_length=80,
        success_threshold=0.48,
        description="Recover enrollment and engagement after a pandemic shock.",
    ),
]


# ---------------------------------------------------------------------------
# LLM policy
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are VIDYA, an AI policymaker for an educational system simulator. "
    "At every step you receive the system state as 13 normalized metrics and "
    "must output 8 intervention intensities, each a float in [0,1], "
    "matching this order: " + ", ".join(ACTION_LABELS) + ". "
    "Reply with ONLY a JSON list of 8 floats, no commentary."
)


def _fallback_action(obs: np.ndarray | None = None) -> np.ndarray:
    """Budget-aware heuristic policy used when no LLM is available.

    Cheap baseline + reactive boosts when key metrics deteriorate, scaled
    down as the budget shrinks so the agent doesn't bankrupt the system.
    """
    base = np.array(
        [0.15, 0.15, 0.20, 0.20, 0.15, 0.30, 0.05, 0.20],
        dtype=np.float32,
    )
    if obs is None or len(obs) < 12:
        return base

    dropout = float(obs[2])
    teacher_retention = float(obs[3])
    student_engagement = float(obs[8])
    budget_norm = float(obs[11])  # budget_remaining / 2_000_000

    # React to crisis signals.
    if dropout > 0.20:
        base[2] = min(1.0, base[2] + 0.25)   # student_scholarship
        base[7] = min(1.0, base[7] + 0.20)   # counseling_programs
    if teacher_retention < 0.55:
        base[1] = min(1.0, base[1] + 0.30)   # teacher_incentive
    if student_engagement < 0.45:
        base[3] = min(1.0, base[3] + 0.20)   # attendance_mandate

    # Scale everything by remaining budget so we never collapse on cost.
    scale = float(np.clip(budget_norm * 1.8, 0.15, 1.0))
    return np.clip(base * scale, 0.0, 1.0).astype(np.float32)


def _parse_action(text: str) -> np.ndarray:
    try:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            return _fallback_action()
        arr = json.loads(match.group(0))
        vec = np.array(arr, dtype=np.float32).flatten()
        if vec.shape[0] != 8:
            return _fallback_action()
        return np.clip(vec, 0.0, 1.0)
    except Exception:
        return _fallback_action()


_PLACEHOLDER_DEFAULTS = {"<your-active-endpoint>", "<your-active-model>", "", None}


class LLMPolicy:
    def __init__(self) -> None:
        # ─────────────────────────────────────────────────────────────────
        # Read the EXACT variables the hackathon's LiteLLM proxy injects.
        # Per their spec:
        #   base_url = os.environ["API_BASE_URL"]
        #   api_key  = os.environ["API_KEY"]
        # Local dev falls back to HF_TOKEN so the same script works
        # against Groq / Fireworks / etc.
        # ─────────────────────────────────────────────────────────────────
        self.base_url = os.environ.get("API_BASE_URL", API_BASE_URL)
        self.model = os.environ.get("MODEL_NAME", MODEL_NAME)
        self.token = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or HF_TOKEN
        self.client = None
        # Treat placeholder-default values as "unset" so the script falls back
        # to the heuristic policy instead of trying to call a fake endpoint.
        self.enabled = (
            self.base_url not in _PLACEHOLDER_DEFAULTS
            and self.model not in _PLACEHOLDER_DEFAULTS
            and self.token not in _PLACEHOLDER_DEFAULTS
        )
        if self.enabled:
            try:
                from openai import OpenAI  # type: ignore
                self.client = OpenAI(base_url=self.base_url, api_key=self.token)
                print(
                    f"[LLM] connected base_url={self.base_url} model={self.model}",
                    flush=True,
                )
            except Exception as e:
                print(f"[LLM] Failed to init OpenAI client: {e}", flush=True)
                self.enabled = False
        else:
            print(
                f"[LLM] disabled — base_url={self.base_url!r} "
                f"model={self.model!r} token_set={bool(self.token)}",
                flush=True,
            )

    def warmup(self) -> bool:
        """Make one tiny call so the LiteLLM proxy registers the key.
        Returns True if the call succeeds, False otherwise. Errors are
        printed loudly to stdout (not swallowed) so they show up in
        validator logs.
        """
        if not self.enabled or self.client is None:
            return False
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with the single word: ready"},
                ],
                temperature=0.0,
                max_tokens=8,
                timeout=30,
            )
            content = (resp.choices[0].message.content or "").strip()
            print(f"[LLM] warmup OK reply={content!r}", flush=True)
            return True
        except Exception as e:
            print(f"[LLM] warmup FAILED: {type(e).__name__}: {e}", flush=True)
            return False

    def act(self, obs: np.ndarray, task: Task) -> np.ndarray:
        if not self.enabled or self.client is None:
            return _fallback_action(obs)
        state_dict = {OBS_LABELS[i]: float(obs[i]) for i in range(len(OBS_LABELS))}
        user_msg = (
            f"Task: {task.description}\n"
            f"Difficulty: {task.difficulty}\n"
            f"State: {json.dumps(state_dict)}\n"
            "Return JSON list of 8 floats."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=128,
                timeout=20,
            )
            return _parse_action(resp.choices[0].message.content or "")
        except Exception as e:
            # Print to STDOUT (not stderr) so the validator's log capture
            # picks this up. Don't swallow silently — the whole point of
            # Phase 2 is that real proxy calls happen.
            print(f"[LLM] call failed: {type(e).__name__}: {e}", flush=True)
            return _fallback_action(obs)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task: Task, policy: LLMPolicy, seed: int = 0) -> Dict:
    env = DropoutCommonsEnv(
        scenario=task.scenario_factory(),
        episode_length=task.episode_length,
    )
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    health_scores: List[float] = []
    collapsed = False
    steps = 0

    print(f"[START] task={task.task_id} difficulty={task.difficulty} "
          f"episode_length={task.episode_length}", flush=True)

    for t in range(task.episode_length):
        action = policy.act(obs, task)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        health = float(info.get("health_score", 0.0))
        health_scores.append(health)
        steps += 1

        print(
            f"[STEP] task={task.task_id} step={t} reward={reward:.4f} "
            f"health={health:.4f} dropout={info.get('dropout_rate', 0):.4f} "
            f"teacher_retention={info.get('teacher_retention', 0):.4f}",
            flush=True,
        )

        if terminated:
            collapsed = True
            break
        if truncated:
            break

    avg_health = float(np.mean(health_scores)) if health_scores else 0.0
    score, passed = task.grade(avg_health, collapsed)

    print(
        f"[END] task={task.task_id} steps={steps} total_reward={total_reward:.4f} "
        f"avg_health={avg_health:.4f} collapsed={int(collapsed)} "
        f"score={score:.4f} passed={int(passed)}",
        flush=True,
    )

    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "steps": steps,
        "total_reward": total_reward,
        "avg_health": avg_health,
        "collapsed": collapsed,
        "score": score,
        "passed": passed,
    }


def main() -> int:
    t0 = time.time()
    policy = LLMPolicy()
    if not policy.enabled:
        print(
            "[LLM] disabled — set API_BASE_URL + API_KEY + MODEL_NAME to enable. "
            "Falling back to heuristic policy.",
            flush=True,
        )
    else:
        # Eagerly call the proxy ONCE so it registers the key. If the call
        # fails, we surface the exact error to stdout (not stderr) so the
        # validator's log capture sees it instead of a silent fallback.
        ok = policy.warmup()
        if not ok:
            print(
                "[LLM] warmup failed — every step will fall back to the "
                "heuristic policy. The proxy will NOT register any hits.",
                flush=True,
            )

    results = []
    for i, task in enumerate(TASKS):
        results.append(run_task(task, policy, seed=42 + i))

    aggregate_score = float(np.mean([r["score"] for r in results]))
    pass_rate = float(np.mean([1.0 if r["passed"] else 0.0 for r in results]))
    elapsed = time.time() - t0

    print(
        f"[END] run=all aggregate_score={aggregate_score:.4f} "
        f"pass_rate={pass_rate:.4f} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
