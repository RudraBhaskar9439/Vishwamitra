"""
Vishwamitra OpenEnv HTTP server
================================

Exposes the DropoutCommonsEnv as an OpenEnv-compatible HTTP API and
mounts the Gradio UI at the same port.

Endpoints
---------
GET  /            → human/agent landing JSON describing the env
POST /reset       → reset the env, return {"observation": [...], "info": {...}}
POST /step        → body {"action": [...]} → 5-tuple JSON
GET  /state       → current observation as JSON
GET  /healthz     → liveness probe (returns "ok")

The Gradio Vishwamitra UI is mounted at /ui.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from env.dropout_env import DropoutCommonsEnv
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from env.scenarios.pandemic_recovery import PandemicRecoveryScenario
from env.scenarios.conflict_zone import ConflictZoneScenario
from env.scenarios.indian_context import IndianContextScenario


SCENARIO_MAP = {
    "funding_cut": FundingCutScenario,
    "teacher_shortage": TeacherShortageScenario,
    "pandemic_recovery": PandemicRecoveryScenario,
    "conflict_zone": ConflictZoneScenario,
    "indian_context": IndianContextScenario,
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    scenario: Optional[str] = None
    episode_length: Optional[int] = None


class StepRequest(BaseModel):
    action: List[float] = Field(..., description="8-dim continuous action vector in [0,1]")


class ResetResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any]


class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    observation: List[float]
    step: int


# ---------------------------------------------------------------------------
# Env singleton
# ---------------------------------------------------------------------------

_env: Optional[DropoutCommonsEnv] = None
_current_scenario_name = "funding_cut"
_current_episode_length = 100


def _build_env(scenario_name: str, episode_length: int) -> DropoutCommonsEnv:
    cls = SCENARIO_MAP.get(scenario_name, FundingCutScenario)
    return DropoutCommonsEnv(scenario=cls(), episode_length=episode_length)


def _get_env() -> DropoutCommonsEnv:
    global _env
    if _env is None:
        _env = _build_env(_current_scenario_name, _current_episode_length)
        _env.reset(seed=0)
    return _env


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / dataclass objects into JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Vishwamitra — DropoutCommons OpenEnv",
    version="0.1.0",
    description=(
        "OpenEnv-compatible HTTP API for the Vishwamitra educational-system "
        "crisis simulator. Built for Meta · PyTorch Hackathon Round 1."
    ),
)


@api.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "vishwamitra-dropout-commons",
            "version": "0.1.0",
            "spec": "openenv",
            "observation_space": {
                "type": "Box",
                "shape": [13],
                "low": 0.0,
                "high": 1.0,
                "dtype": "float32",
            },
            "action_space": {
                "type": "Box",
                "shape": [8],
                "low": 0.0,
                "high": 1.0,
                "dtype": "float32",
            },
            "endpoints": ["/reset", "/step", "/state", "/healthz", "/ui"],
            "scenarios": list(SCENARIO_MAP.keys()),
        }
    )


@api.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@api.post("/reset", response_model=ResetResponse)
def reset(body: Optional[ResetRequest] = None) -> ResetResponse:
    """Reset the environment. Body is optional — empty POST works."""
    global _env, _current_scenario_name, _current_episode_length
    seed = None
    if body is not None:
        if body.scenario and body.scenario in SCENARIO_MAP:
            _current_scenario_name = body.scenario
        if body.episode_length and body.episode_length > 0:
            _current_episode_length = int(body.episode_length)
        seed = body.seed
    _env = _build_env(_current_scenario_name, _current_episode_length)
    obs, info = _env.reset(seed=seed)
    return ResetResponse(
        observation=[float(v) for v in np.asarray(obs).tolist()],
        info=_to_jsonable(info),
    )


@api.post("/step", response_model=StepResponse)
def step(body: StepRequest) -> StepResponse:
    env = _get_env()
    action = np.asarray(body.action, dtype=np.float32)
    if action.shape != (8,):
        # pad / truncate so the request never crashes the env
        flat = action.flatten()
        if flat.size < 8:
            flat = np.concatenate([flat, np.zeros(8 - flat.size, dtype=np.float32)])
        action = flat[:8]
    action = np.clip(action, 0.0, 1.0)
    obs, reward, terminated, truncated, info = env.step(action)
    return StepResponse(
        observation=[float(v) for v in np.asarray(obs).tolist()],
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=_to_jsonable(info),
    )


@api.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    env = _get_env()
    if env.state is None:
        env.reset(seed=0)
    return StateResponse(
        observation=[float(v) for v in env.state.to_obs_array().tolist()],
        step=int(env.state.step),
    )


# ---------------------------------------------------------------------------
# Mount the Gradio UI at /ui (best-effort — server still starts if Gradio
# import fails or LLM creds are missing).
# ---------------------------------------------------------------------------

try:
    import gradio as gr  # type: ignore
    import app as _vish_app  # noqa: E402
    _gradio_demo = _vish_app.create_spaces_demo()
    api = gr.mount_gradio_app(api, _gradio_demo, path="/ui")
except Exception as _e:  # noqa: BLE001
    print(f"[server] Gradio UI not mounted: {_e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server:api", host="0.0.0.0", port=port, log_level="info")
