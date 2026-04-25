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

from fastapi.middleware.cors import CORSMiddleware


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

# CORS for local Vite dev server (port 5173) and any frontend during dev.
try:
    from fastapi.middleware.cors import CORSMiddleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:  # noqa: BLE001
    pass

# Mount the Vishwamitra swarm-deliberation router (loads roles.yaml, manages
# 4 swarms x 3 personas, returns ResonanceReport).
try:
    from server.swarm_routes import router as _swarm_router
    api.include_router(_swarm_router)
except Exception as _swarm_err:  # noqa: BLE001
    print(f"[server] swarm router not mounted: {_swarm_err}")


_OPENENV_INFO = {
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
    "endpoints": ["/reset", "/step", "/state", "/healthz", "/info"],
    "scenarios": list(SCENARIO_MAP.keys()),
}


api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Simulator JSON API (wraps the Gradio VIDYADemo handlers so a React frontend
# can call them over plain HTTP).
# ---------------------------------------------------------------------------

_demo_instance = None


def _get_demo():
    global _demo_instance
    if _demo_instance is None:
        import app as _vish_app  # noqa: E402
        _demo_instance = _vish_app.VIDYADemo()
    return _demo_instance


class LoadPolicyRequest(BaseModel):
    model_type: str = "meta_rl"


class CreateScenarioRequest(BaseModel):
    crisis_text: str = ""
    difficulty: str = "medium"
    initial_budget: float = 70
    teacher_retention: float = 75
    enrollment_rate: float = 85


class RunSimRequest(BaseModel):
    selected_agents: List[str] = ["Student", "Teacher", "Administrator", "Policymaker"]
    n_steps: int = 100
    use_interventions: bool = True


class FeedbackRequest(BaseModel):
    rating: int = 3
    comment: str = ""


class CompareRequest(BaseModel):
    n_steps: int = 50


def _figure_to_json(fig):
    if fig is None:
        return None
    try:
        import json as _json
        import plotly  # type: ignore
        return _json.loads(plotly.io.to_json(fig))
    except Exception:
        return None


@api.post("/api/load_policy")
def api_load_policy(body: LoadPolicyRequest):
    msg = _get_demo().load_model(body.model_type)
    return {"status": msg}


@api.post("/api/create_scenario")
def api_create_scenario(body: CreateScenarioRequest):
    msg = _get_demo().create_scenario(
        body.crisis_text,
        body.difficulty,
        float(body.initial_budget),
        float(body.teacher_retention),
        float(body.enrollment_rate),
    )
    return {"status": msg}


@api.post("/api/run_simulation")
def api_run_simulation(body: RunSimRequest):
    perspectives, verdict, status, traj, metrics, interv = _get_demo().run_simulation(
        body.selected_agents, int(body.n_steps), bool(body.use_interventions)
    )
    return {
        "perspectives_md": perspectives,
        "verdict_md": verdict,
        "status_md": status,
        "trajectory_plot": _figure_to_json(traj),
        "metrics_plot": _figure_to_json(metrics),
        "intervention_plot": _figure_to_json(interv),
    }


@api.post("/api/feedback")
def api_feedback(body: FeedbackRequest):
    perspectives, verdict, status, traj, metrics, interv = _get_demo().submit_inline_feedback(
        int(body.rating), body.comment
    )
    return {
        "perspectives_md": perspectives,
        "verdict_md": verdict,
        "status_md": status,
        "trajectory_plot": _figure_to_json(traj),
        "metrics_plot": _figure_to_json(metrics),
        "intervention_plot": _figure_to_json(interv),
    }


@api.post("/api/compare")
def api_compare(body: CompareRequest):
    report, fig = _get_demo().compare_policies(int(body.n_steps))
    return {"report": report, "plot": _figure_to_json(fig)}


@api.get("/api/config")
def api_config():
    """Runtime config for the React frontend (Hume creds, etc).
    Reads from environment so Hugging Face Spaces secrets work without
    rebuilding the static bundle."""
    return {
        "hume_api_key": os.environ.get("HUME_API_KEY", ""),
        "hume_config_id": os.environ.get("HUME_CONFIG_ID", ""),
    }


@api.get("/info")
def info() -> JSONResponse:
    return JSONResponse(_OPENENV_INFO)


@api.get("/openenv")
def openenv_alias() -> JSONResponse:
    return JSONResponse(_OPENENV_INFO)


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
    # Gradio lives at /gradio so the React frontend can own /.
    api = gr.mount_gradio_app(api, _gradio_demo, path="/gradio")
except Exception as _e:  # noqa: BLE001
    print(f"[server] Gradio UI not mounted: {_e}")


# ---------------------------------------------------------------------------
# Static React frontend (built into frontend/dist by the Dockerfile).
# Mounted last so explicit /api/* and OpenEnv routes still take precedence.
# ---------------------------------------------------------------------------
try:
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    _frontend_dist = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "dist",
    )
    if os.path.isdir(_frontend_dist):
        api.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="assets")

        @api.get("/")
        def _root():
            return FileResponse(os.path.join(_frontend_dist, "index.html"))

        @api.get("/{full_path:path}")
        def _spa_fallback(full_path: str):
            # Serve real files if they exist, otherwise fall back to index.html
            candidate = os.path.join(_frontend_dist, full_path)
            if os.path.isfile(candidate):
                return FileResponse(candidate)
            return FileResponse(os.path.join(_frontend_dist, "index.html"))

        print(f"[server] React frontend mounted from {_frontend_dist}")
    else:
        print(f"[server] React dist not found at {_frontend_dist} — frontend not served")
except Exception as _e:  # noqa: BLE001
    print(f"[server] Static frontend not mounted: {_e}")


def main() -> None:
    """Console-script entry point: launches uvicorn on $PORT (default 7860).
    Importable as `server.app:main`.
    """
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:api", host="0.0.0.0", port=port, log_level="info")


# Backwards-compat alias
serve = main


if __name__ == "__main__":
    main()
