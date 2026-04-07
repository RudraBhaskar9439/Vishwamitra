"""
VIDYA Pre-Submission Validation
================================

Self-checks the repository against OpenEnv Round 1 requirements.
Run with:  python validate.py
Exits non-zero on any failure.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
FAILURES: list[str] = []
PASSES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSES.append(name)
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name} — {detail}")
        print(f"  FAIL  {name}  {detail}")


def main() -> int:
    print("VIDYA Pre-Submission Validation\n" + "=" * 40)

    # 1. Required files
    print("\n[1] Required files")
    for f in [
        "inference.py", "Dockerfile", "openenv.yaml",
        "README.md", "requirements.txt", "app.py",
    ]:
        check(f"file exists: {f}", (ROOT / f).is_file())

    # 2. Importability
    print("\n[2] Imports")
    try:
        from env.dropout_env import DropoutCommonsEnv  # noqa
        from env.openenv_compat import VidyaOpenEnvWrapper  # noqa
        from env.scenarios.funding_cut import FundingCutScenario  # noqa
        from env.scenarios.teacher_shortage import TeacherShortageScenario  # noqa
        from env.scenarios.pandemic_recovery import PandemicRecoveryScenario  # noqa
        check("env imports", True)
    except Exception as e:
        check("env imports", False, str(e))

    # 3. Env reset/step contract
    print("\n[3] Env reset()/step() contract")
    try:
        from env.dropout_env import DropoutCommonsEnv
        env = DropoutCommonsEnv(episode_length=10)
        obs, info = env.reset(seed=0)
        check("reset returns (obs, info)", obs is not None and isinstance(info, dict))
        check("obs shape == (13,)", tuple(obs.shape) == (13,))
        import numpy as np
        action = np.full(8, 0.5, dtype=np.float32)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        check("step returns 5-tuple", True)
        check("reward is float", isinstance(reward, float))
    except Exception as e:
        check("env contract", False, str(e))

    # 4. inference.py importable & has main
    print("\n[4] inference.py")
    try:
        sys.path.insert(0, str(ROOT))
        inf = importlib.import_module("inference")
        check("inference.main exists", hasattr(inf, "main"))
        check("inference.TASKS has >=3", len(getattr(inf, "TASKS", [])) >= 3)
        diffs = {t.difficulty for t in inf.TASKS}
        check("tasks cover easy/medium/hard",
              {"easy", "medium", "hard"}.issubset(diffs))
    except Exception as e:
        check("inference.py import", False, str(e))

    # 5. openenv.yaml parseable
    print("\n[5] openenv.yaml")
    try:
        import yaml  # type: ignore
        with open(ROOT / "openenv.yaml") as f:
            cfg = yaml.safe_load(f)
        check("openenv.yaml parses", isinstance(cfg, dict))
        check("declares >=3 tasks", len(cfg.get("tasks", [])) >= 3)
        check("declares env_vars", set(cfg.get("env_vars", [])) >=
              {"API_BASE_URL", "MODEL_NAME", "HF_TOKEN"})
    except ModuleNotFoundError:
        print("  SKIP  PyYAML not installed; skipping yaml parse")
    except Exception as e:
        check("openenv.yaml", False, str(e))

    # 6. Dockerfile sanity
    print("\n[6] Dockerfile")
    try:
        text = (ROOT / "Dockerfile").read_text()
        check("Dockerfile has FROM", "FROM " in text)
        check("Dockerfile copies requirements", "requirements.txt" in text)
        check("Dockerfile installs deps", "pip install" in text)
    except Exception as e:
        check("Dockerfile", False, str(e))

    # Summary
    print("\n" + "=" * 40)
    print(f"Passed: {len(PASSES)}   Failed: {len(FAILURES)}")
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll pre-submission checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
