"""
Dynamic weight + model allocator for the swarm-of-swarms.

The WeightAllocator inspects the current system state, computes a per-role
"attention" score that captures how much that role's lens matters in the
current crisis, and returns a RouterDecision per role specifying:

  - Which LLM model to use for that role's swarm
  - What verdict weight to apply when aggregating across swarms

Two models are supported by default (both via Groq):
  - llama-3.3-70b-versatile  (heavy, 12K TPM, used for high-attention roles)
  - llama-3.1-8b-instant     (light,  6K TPM, used for routine roles)

Allocation modes:
  - "auto"   : every role uses the heuristic
  - "manual" : every role uses operator-supplied overrides
  - "mixed"  : some roles overridden, others fall back to heuristic

The orchestrator never touches the resonance metric itself — verdict
weights modulate the aggregated action vector (whose lens has more pull on
the final recommendation), but resonance still reflects raw disagreement
across the four lenses. We do this on purpose: dissent is the signal we
care about, and we don't want operator-set weights to cosmetically erase
genuine stakeholder disagreement.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


HEAVY_MODEL_DEFAULT = "llama-3.3-70b-versatile"
LIGHT_MODEL_DEFAULT = "llama-3.1-8b-instant"

ROLE_ORDER = ("student", "teacher", "admin", "policymaker")

# Canonical pressure names. Keys must match the fit_signals declared on
# each persona in roles.yaml. Higher pressure value = more crisis on that
# dimension. All values clamped to [0, 1].
PRESSURE_NAMES = (
    "dropout", "burnout", "budget", "retention", "enrollment",
    "engagement", "attendance", "resource", "class_size", "trust",
)


def compute_state_pressures(state: dict[str, Any]) -> dict[str, float]:
    """Convert a raw system state into normalised pressure signals in [0, 1].

    Used by both the L1 WeightAllocator (per-role attention) and the L2
    PersonaAllocator (per-persona fit). Each pressure is engineered to
    point in the same direction: higher = worse.
    """
    def g(key: str, default: float = 0.0) -> float:
        v = state.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    clamp01 = lambda x: max(0.0, min(1.0, x))

    dropout       = g("dropout_rate")
    burnout       = g("teacher_burnout")
    retention     = g("teacher_retention", 0.7)
    budget        = g("budget_remaining", 1_000_000.0)
    enrollment    = g("enrollment_rate", 0.7)
    engagement    = g("student_engagement", 0.6)
    attendance    = g("attendance_rate", 0.7)
    resources     = g("resource_allocation", 0.6)
    class_size    = g("avg_class_size", 35.0)
    trust         = g("trust_score", 1.0)

    return {
        "dropout":     clamp01(dropout / 0.4),
        "burnout":     clamp01(burnout / 0.8),
        "budget":      clamp01(1.0 - budget / 1_000_000.0),
        "retention":   clamp01(1.0 - retention / 0.7),
        "enrollment":  clamp01(1.0 - enrollment / 0.6),
        "engagement":  clamp01(1.0 - engagement / 0.6),
        "attendance":  clamp01(1.0 - attendance / 0.7),
        "resource":    clamp01(1.0 - resources / 0.5),
        "class_size":  clamp01((class_size - 30.0) / 30.0),
        "trust":       clamp01(1.0 - trust),
    }


@dataclass
class RouterDecision:
    role: str
    model: str
    verdict_weight: float
    attention_score: float           # in [0, 1]
    source: str                      # "auto" | "manual"
    reason: str                      # human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorPlan:
    decisions: dict[str, RouterDecision]   # keyed by role
    crisis_signal: float                   # max attention across roles
    mode: str                              # "auto" | "manual" | "mixed"
    heavy_model: str
    light_model: str
    attention_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "crisis_signal": self.crisis_signal,
            "mode": self.mode,
            "heavy_model": self.heavy_model,
            "light_model": self.light_model,
            "attention_threshold": self.attention_threshold,
        }

    def model_for(self, role: str) -> str | None:
        d = self.decisions.get(role)
        return d.model if d else None

    def weight_for(self, role: str) -> float:
        d = self.decisions.get(role)
        return d.verdict_weight if d else 1.0


class WeightAllocator:
    """State-driven router. See module docstring for behavior."""

    def __init__(
        self,
        heavy_model: str = HEAVY_MODEL_DEFAULT,
        light_model: str = LIGHT_MODEL_DEFAULT,
        attention_threshold: float = 0.6,
        heavy_verdict_weight: float = 1.5,
        light_verdict_weight: float = 1.0,
    ):
        self.heavy_model = heavy_model
        self.light_model = light_model
        self.attention_threshold = attention_threshold
        self.heavy_verdict_weight = heavy_verdict_weight
        self.light_verdict_weight = light_verdict_weight

    # ------------------------- public API -------------------------
    def allocate(
        self,
        state: dict[str, Any],
        manual_overrides: dict[str, Any] | None = None,
    ) -> OrchestratorPlan:
        """Return a plan covering all four roles.

        manual_overrides shape (all keys optional):
            {
              "mode": "auto" | "manual",   # informational; not strictly required
              "roles": {
                "student": {"model": "...", "verdict_weight": 1.5},
                "teacher": {...},
                ...
              }
            }
        Any role not present in `roles` falls back to the heuristic.
        """
        attention = self._compute_role_attention(state or {})
        crisis_signal = max(attention.values()) if attention else 0.0

        overrides_block = (manual_overrides or {}).get("roles") or {}
        any_override = bool(overrides_block)
        all_override = (
            any_override
            and all(role in overrides_block for role in ROLE_ORDER)
        )
        mode = "manual" if all_override else ("mixed" if any_override else "auto")

        decisions: dict[str, RouterDecision] = {}
        for role in ROLE_ORDER:
            score = float(attention.get(role, 0.0))
            override = overrides_block.get(role)
            if override:
                model = str(
                    override.get("model")
                    or self._auto_model(score)
                )
                try:
                    weight = float(override.get(
                        "verdict_weight", self._auto_weight(score)
                    ))
                except (TypeError, ValueError):
                    weight = self._auto_weight(score)
                source = "manual"
                reason = f"Operator override · attention {score:.2f}"
            else:
                model = self._auto_model(score)
                weight = self._auto_weight(score)
                source = "auto"
                reason = self._auto_reason(score)
            decisions[role] = RouterDecision(
                role=role,
                model=model,
                verdict_weight=weight,
                attention_score=score,
                source=source,
                reason=reason,
            )

        return OrchestratorPlan(
            decisions=decisions,
            crisis_signal=crisis_signal,
            mode=mode,
            heavy_model=self.heavy_model,
            light_model=self.light_model,
            attention_threshold=self.attention_threshold,
        )

    # ------------------------- heuristics -------------------------
    def _auto_model(self, score: float) -> str:
        return self.heavy_model if score >= self.attention_threshold else self.light_model

    def _auto_weight(self, score: float) -> float:
        return (
            self.heavy_verdict_weight
            if score >= self.attention_threshold
            else self.light_verdict_weight
        )

    def _auto_reason(self, score: float) -> str:
        if score >= self.attention_threshold:
            return (
                f"High pressure (attention {score:.2f} ≥ "
                f"{self.attention_threshold:.2f}) — heavyweight model "
                f"with {self.heavy_verdict_weight:.1f}× verdict weight"
            )
        return (
            f"Routine (attention {score:.2f} < "
            f"{self.attention_threshold:.2f}) — lightweight model "
            f"with {self.light_verdict_weight:.1f}× verdict weight"
        )

    def _compute_role_attention(self, state: dict[str, Any]) -> dict[str, float]:
        """Map the current state to a per-role attention score in [0, 1].

        Each role's score is the max of the pressures most relevant to that
        role's lens. We use max (not mean) so a single sharp pressure can
        promote a role to heavyweight without being diluted by other-domain
        calmness.
        """
        p = compute_state_pressures(state)
        return {
            "student":      max(p["dropout"], p["enrollment"], p["engagement"], p["attendance"]),
            "teacher":      max(p["burnout"], p["retention"], p["class_size"]),
            "admin":        max(p["budget"], p["resource"]),
            "policymaker":  max(p["dropout"], p["budget"], p["trust"], p["retention"] * 0.8),
        }


# ============================================================================
# L2 ORCHESTRATOR — per-persona allocation within a swarm
# ============================================================================

@dataclass
class PersonaFitDecision:
    """How well one persona's declared fit_signals match the current state."""
    persona_id: str
    persona_name: str
    role: str
    weight: float                             # final weight applied at aggregation
    fit_score: float                          # raw fit score in [0, 1]
    matched_signals: dict[str, float]         # signal -> contribution (sig_weight × pressure)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersonaAllocator:
    """L2 orchestrator: allocates within-swarm verdict weight per persona.

    For each persona, scores its `fit_signals` against the current state's
    pressures (the same canonical pressures used by the L1 WeightAllocator).
    A persona declaring fit_signals = {"budget": 0.9} contributes more
    weight when budget pressure is high.

    Final weight is mapped from fit_score [0,1] into [base_floor, boost],
    so even an "off-topic" persona keeps a non-zero voice — the swarm
    still hears all 3 lenses, just at different volumes.
    """

    def __init__(
        self,
        base_floor: float = 0.3,
        boost_high_fit: float = 1.5,
    ):
        self.base_floor = base_floor
        self.boost_high_fit = boost_high_fit

    def allocate(
        self,
        personas: list,                       # list[Persona] — kept loose to avoid circular import
        state: dict[str, Any],
    ) -> list[PersonaFitDecision]:
        pressures = compute_state_pressures(state or {})
        out: list[PersonaFitDecision] = []
        for p in personas:
            signals = getattr(p, "fit_signals", None) or {}
            if not signals:
                # No fit_signals declared → flat weight = boost mid-point.
                neutral = (self.base_floor + self.boost_high_fit) / 2.0
                out.append(PersonaFitDecision(
                    persona_id=p.id,
                    persona_name=p.name,
                    role=p.role,
                    weight=neutral,
                    fit_score=0.5,
                    matched_signals={},
                ))
                continue

            total = 0.0
            wsum = 0.0
            matched: dict[str, float] = {}
            for sig_name, sig_weight in signals.items():
                pressure = float(pressures.get(sig_name, 0.0))
                sw = float(sig_weight)
                contribution = pressure * sw
                total += contribution
                wsum += sw
                if pressure > 0.05:           # only report signals that fired
                    matched[sig_name] = round(contribution, 3)

            fit_score = (total / wsum) if wsum > 0 else 0.0
            fit_score = max(0.0, min(1.0, fit_score))
            weight = self.base_floor + fit_score * (self.boost_high_fit - self.base_floor)

            out.append(PersonaFitDecision(
                persona_id=p.id,
                persona_name=p.name,
                role=p.role,
                weight=weight,
                fit_score=fit_score,
                matched_signals=matched,
            ))
        return out

    def allocate_dict(
        self,
        personas: list,
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Convenience: return persona_id → weight mapping only."""
        return {d.persona_id: d.weight for d in self.allocate(personas, state)}


__all__ = [
    "WeightAllocator",
    "RouterDecision",
    "OrchestratorPlan",
    "PersonaAllocator",
    "PersonaFitDecision",
    "compute_state_pressures",
    "PRESSURE_NAMES",
    "ROLE_ORDER",
    "HEAVY_MODEL_DEFAULT",
    "LIGHT_MODEL_DEFAULT",
]
