"""
check_hf_space.py — Pre-submission sanity check for the Vishwamitra HF Space.

Hits every endpoint a hackathon judge will probe and confirms each one
returns the right shape. Exit code 0 if everything passes, 1 otherwise.

Usage:
    python check_hf_space.py
    python check_hf_space.py --url https://your-other-space.hf.space
    python check_hf_space.py --include-deliberate     # also hit the LLM swarm

Run this BEFORE you submit. Cold-start the Space will take 30-60s the first
time; the script auto-retries during warm-up.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
except ImportError:
    print("Install requests:  pip install requests")
    sys.exit(2)


DEFAULT_URL = "https://rudra9439-vidya-meta-rl.hf.space"


# Terminal colors — pure ANSI, no deps
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    AMBER  = "\033[93m"
    DIM    = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def ok(msg: str) -> None:    print(f"  {C.GREEN}✓ PASS{C.RESET}  {msg}")
def fail(msg: str) -> None:  print(f"  {C.RED}✗ FAIL{C.RESET}  {msg}")
def warn(msg: str) -> None:  print(f"  {C.AMBER}⚠ WARN{C.RESET}  {msg}")
def info(msg: str) -> None:  print(f"  {C.DIM}·{C.RESET}      {msg}")


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""
    ms: float = 0.0


@dataclass
class Suite:
    base: str
    timeout: float = 30.0
    results: list[Result] = field(default_factory=list)

    # ----------------------- HTTP helpers -----------------------
    def _request(self, method: str, path: str, **kw) -> requests.Response | None:
        url = self.base.rstrip("/") + path
        try:
            t0 = time.time()
            resp = requests.request(method, url, timeout=self.timeout, **kw)
            ms = (time.time() - t0) * 1000.0
            return resp, ms
        except requests.RequestException as e:
            return None, str(e)

    def _record(self, name: str, passed: bool, detail: str = "", ms: float = 0.0) -> None:
        self.results.append(Result(name, passed, detail, ms))
        line = f"({ms:>6.0f} ms)  {detail}" if ms else detail
        (ok if passed else fail)(f"{name:<32}  {line}")

    # ----------------------- individual checks -----------------------
    def warmup(self, max_attempts: int = 6) -> bool:
        """HF Spaces sleep — the first request after a long idle can take
        30-60s while the container boots. Retry until it responds."""
        print(f"{C.BOLD}Warming up Space at {self.base} ...{C.RESET}")
        for attempt in range(1, max_attempts + 1):
            r, ms = self._request("GET", "/healthz")
            if r is None:
                info(f"attempt {attempt}/{max_attempts}: network error ({ms})")
                time.sleep(5)
                continue
            if r.status_code in (200, 404):  # 404 means it's serving but no /healthz
                info(f"attempt {attempt}/{max_attempts}: HTTP {r.status_code} after {ms:.0f}ms — Space is awake")
                return True
            info(f"attempt {attempt}/{max_attempts}: HTTP {r.status_code} — waiting...")
            time.sleep(5)
        return False

    def check_health(self) -> None:
        r, ms = self._request("GET", "/healthz")
        if r is None:
            self._record("GET /healthz", False, f"network error: {ms}")
            return
        if r.status_code == 200:
            self._record("GET /healthz", True, f"HTTP 200, body={r.text.strip()[:40]!r}", ms)
        elif r.status_code == 404:
            self._record("GET /healthz", False, "endpoint missing — server hasn't mounted /healthz")
        else:
            self._record("GET /healthz", False, f"HTTP {r.status_code}")

    def check_openenv_info(self) -> None:
        # Most OpenEnv servers expose / and /openenv with the manifest
        for path in ("/", "/openenv"):
            r, ms = self._request("GET", path)
            if r is None:
                self._record(f"GET {path}", False, "network error")
                continue
            if r.status_code != 200:
                self._record(f"GET {path}", False, f"HTTP {r.status_code}")
                continue
            try:
                data = r.json()
            except Exception:
                self._record(f"GET {path}", False, "response is not JSON")
                continue

            required = {"name", "observation_space", "action_space"}
            missing = required - set(data.keys())
            if missing:
                self._record(f"GET {path}", False, f"missing keys: {sorted(missing)}", ms)
            else:
                obs_shape = (data.get("observation_space") or {}).get("shape")
                act_shape = (data.get("action_space") or {}).get("shape")
                self._record(
                    f"GET {path}", True,
                    f"name={data.get('name')!r}  obs={obs_shape}  act={act_shape}",
                    ms,
                )

    def check_reset(self) -> dict | None:
        r, ms = self._request(
            "POST", "/reset",
            json={"scenario": "funding_cut", "episode_length": 100},
            headers={"Content-Type": "application/json"},
        )
        if r is None:
            self._record("POST /reset", False, "network error")
            return None
        if r.status_code != 200:
            self._record("POST /reset", False, f"HTTP {r.status_code}: {r.text[:120]}")
            return None
        try:
            data = r.json()
        except Exception:
            self._record("POST /reset", False, "response is not JSON")
            return None

        if "observation" not in data:
            self._record("POST /reset", False, "missing 'observation' in response")
            return None

        obs_len = len(data["observation"])
        if obs_len not in (13, 14):  # 13 nominal, 14 if data_integrity also exposed
            self._record("POST /reset", False, f"observation length {obs_len}, expected 13 or 14")
        else:
            self._record("POST /reset", True, f"observation length {obs_len}", ms)
        return data

    def check_step(self) -> None:
        action = [0.5, 0.5, 0.3, 0.1, 0.4, 0.2, 0.0, 0.3]
        r, ms = self._request(
            "POST", "/step",
            json={"action": action},
            headers={"Content-Type": "application/json"},
        )
        if r is None:
            self._record("POST /step", False, "network error")
            return
        if r.status_code != 200:
            self._record("POST /step", False, f"HTTP {r.status_code}: {r.text[:120]}")
            return
        try:
            data = r.json()
        except Exception:
            self._record("POST /step", False, "response is not JSON")
            return

        # Gym-style 5-tuple keys
        required = {"observation", "reward", "terminated", "truncated"}
        missing = required - set(data.keys())
        if missing:
            self._record("POST /step", False, f"missing keys: {sorted(missing)}")
            return
        self._record(
            "POST /step", True,
            f"reward={float(data['reward']):+.3f}  terminated={data['terminated']}",
            ms,
        )

    def check_state(self) -> None:
        r, ms = self._request("GET", "/state")
        if r is None:
            self._record("GET /state", False, "network error")
            return
        if r.status_code == 404:
            warn("GET /state                       not implemented — that's OK, optional")
            return
        if r.status_code != 200:
            self._record("GET /state", False, f"HTTP {r.status_code}")
            return
        try:
            data = r.json()
            if "observation" in data or isinstance(data, list):
                self._record("GET /state", True, "returned current observation", ms)
            else:
                self._record("GET /state", False, "no 'observation' field in response")
        except Exception:
            self._record("GET /state", False, "response is not JSON")

    def check_swarm_info(self) -> None:
        r, ms = self._request("GET", "/swarms/info")
        if r is None:
            self._record("GET /swarms/info", False, "network error")
            return
        if r.status_code == 404:
            warn("GET /swarms/info                 endpoint missing — swarm router not mounted")
            return
        if r.status_code != 200:
            self._record("GET /swarms/info", False, f"HTTP {r.status_code}")
            return
        try:
            data = r.json()
            roles = data.get("roles", {})
            n_personas = sum(len(v) if isinstance(v, list) else 0 for v in roles.values())
            self._record(
                "GET /swarms/info", True,
                f"model={data.get('model','?')}  roles={list(roles.keys())}  personas={n_personas}",
                ms,
            )
        except Exception as e:
            self._record("GET /swarms/info", False, f"bad JSON: {e}")

    def check_deliberate(self) -> None:
        """Optional — costs a swarm round on the configured LLM provider."""
        body = {
            "state": {
                "enrollment_rate": 0.62, "attendance_rate": 0.55,
                "dropout_rate": 0.28, "teacher_retention": 0.71,
                "teacher_burnout": 0.72, "student_engagement": 0.50,
                "resource_allocation": 0.40, "avg_class_size": 48,
                "budget_remaining": 420000.0, "trust_score": 0.55,
                "step": 12,
            },
            "scenario": "smoke test from check_hf_space.py",
        }
        info("calling /swarms/deliberate (this hits the LLM, may take 10-30 s)...")
        # bigger timeout for this one
        old_to = self.timeout
        self.timeout = 90.0
        try:
            r, ms = self._request(
                "POST", "/swarms/deliberate",
                json=body, headers={"Content-Type": "application/json"},
            )
        finally:
            self.timeout = old_to
        if r is None:
            self._record("POST /swarms/deliberate", False, "network error")
            return
        if r.status_code != 200:
            self._record(
                "POST /swarms/deliberate", False,
                f"HTTP {r.status_code}: {r.text[:120]}",
            )
            return
        try:
            data = r.json()
            n_swarms = len(data.get("swarm_verdicts", []))
            n_flags = len(data.get("dissonance_flags", []))
            self._record(
                "POST /swarms/deliberate", True,
                f"{n_swarms} swarms · {n_flags} dissonance flags",
                ms,
            )
        except Exception as e:
            self._record("POST /swarms/deliberate", False, f"bad JSON: {e}")

    # ----------------------- runner -----------------------
    def run(self, include_deliberate: bool) -> bool:
        if not self.warmup():
            print(f"\n{C.RED}Space failed to warm up after multiple retries.{C.RESET}")
            print("Check the Space logs at https://huggingface.co/spaces/rudra9439/vidya-meta-rl/logs")
            return False
        print()
        print(f"{C.BOLD}Running endpoint checks...{C.RESET}")
        self.check_health()
        self.check_openenv_info()
        self.check_reset()
        self.check_step()
        self.check_state()
        self.check_swarm_info()
        if include_deliberate:
            self.check_deliberate()
        return all(r.passed for r in self.results)

    def summary(self) -> None:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        avg_ms = sum(r.ms for r in self.results) / max(1, total)
        print()
        print("─" * 60)
        if passed == total:
            print(f"{C.GREEN}{C.BOLD}ALL {total} CHECKS PASSED{C.RESET}  ·  avg {avg_ms:.0f} ms")
            print("Your Space is ready for submission. ✨")
        else:
            print(f"{C.RED}{C.BOLD}{total - passed}/{total} CHECKS FAILED{C.RESET}")
            print("\nFailing checks:")
            for r in self.results:
                if not r.passed:
                    print(f"  · {r.name}: {r.detail}")
            print(f"\n{C.AMBER}Fix these before submitting.{C.RESET}")
        print("─" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Base URL of the Space (default: {DEFAULT_URL})")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-request timeout in seconds")
    p.add_argument("--include-deliberate", action="store_true",
                   help="Also exercise POST /swarms/deliberate (uses LLM tokens, slow)")
    args = p.parse_args()

    suite = Suite(base=args.url, timeout=args.timeout)
    all_passed = suite.run(include_deliberate=args.include_deliberate)
    suite.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
