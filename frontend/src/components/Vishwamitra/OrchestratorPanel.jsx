import { useEffect, useRef, useState } from 'react'
import {
  previewPlan,
  ORCHESTRATOR_ROLES,
  ORCHESTRATOR_MODELS,
} from '../../lib/swarm-api.js'
import { ROLE_COLORS } from './SwarmGraph.jsx'

// Debounce a value — used to throttle preview-plan calls when sliders move.
function useDebounced(value, delay = 250) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return v
}

function bar(value, width = 80) {
  const pct = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)) * 100
  return (
    <span className="vm-orch-bar" style={{ width }}>
      <span className="fill" style={{ width: `${pct}%` }} />
    </span>
  )
}

export default function OrchestratorPanel({
  state,
  mode,                  // 'auto' | 'manual'
  onModeChange,
  overrides,             // { [role]: { model, verdict_weight } }
  onOverridesChange,
}) {
  const debouncedState = useDebounced(state, 250)
  const debouncedOverrides = useDebounced(overrides, 250)
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState(null)
  const reqId = useRef(0)

  // Re-preview whenever the (debounced) state OR overrides change.
  useEffect(() => {
    const myId = ++reqId.current
    setLoading(true); setErr(null)
    const overridePayload = mode === 'manual' && Object.keys(overrides || {}).length
      ? { mode: 'manual', roles: overrides }
      : null
    previewPlan({ state: debouncedState, orchestratorOverrides: overridePayload })
      .then((p) => {
        if (myId === reqId.current) setPlan(p)
      })
      .catch((e) => {
        if (myId === reqId.current) setErr(String(e.message || e))
      })
      .finally(() => {
        if (myId === reqId.current) setLoading(false)
      })
  }, [debouncedState, debouncedOverrides, mode])

  const decisions = plan?.decisions || {}
  const crisis = plan?.crisis_signal ?? 0

  function setRoleOverride(role, key, value) {
    onOverridesChange({
      ...overrides,
      [role]: { ...(overrides?.[role] || {}), [key]: value },
    })
  }

  function clearRoleOverride(role) {
    const next = { ...(overrides || {}) }
    delete next[role]
    onOverridesChange(next)
  }

  function clearAllOverrides() {
    onOverridesChange({})
  }

  return (
    <div className="vm-orchestrator">
      <div className="vm-orch-head">
        <span className="vm-orch-title">▸ ORCHESTRATOR</span>
        <span
          className="vm-orch-l2-badge"
          title="L2 PersonaAllocator: per-persona weights computed from fit_signals × state pressures, applied during within-swarm aggregation."
        >L2 ACTIVE</span>
        <span className="vm-orch-modeswitch">
          <button
            className={mode === 'auto' ? 'on' : ''}
            onClick={() => onModeChange('auto')}
            title="Heuristic: state-vector → model + weight per role"
          >AUTO</button>
          <button
            className={mode === 'manual' ? 'on' : ''}
            onClick={() => onModeChange('manual')}
            title="Pin model and weight per role manually"
          >MANUAL</button>
        </span>

        <span className="vm-orch-spacer" />

        <span className="vm-orch-crisis">
          CRISIS {bar(crisis, 90)} {(crisis * 100).toFixed(0)}%
        </span>

        {mode === 'manual' && Object.keys(overrides || {}).length > 0 && (
          <button className="vm-orch-clear" onClick={clearAllOverrides}>
            reset
          </button>
        )}
      </div>

      <div className="vm-orch-rows">
        {ORCHESTRATOR_ROLES.map((role) => {
          const d = decisions[role] || {}
          const color = ROLE_COLORS[role]
          const isHeavy = d.model === plan?.heavy_model
          const isManual = mode === 'manual'
          const ov = (overrides || {})[role] || {}
          const currentModel = ov.model || d.model || ''
          const currentWeight = ov.verdict_weight ?? d.verdict_weight ?? 1.0

          return (
            <div key={role} className={`vm-orch-row ${d.source === 'manual' ? 'manual' : ''}`}>
              <span className="dot" style={{ background: color }} />
              <span className="role">{role}</span>

              {isManual ? (
                <select
                  className="vm-orch-select"
                  value={currentModel}
                  onChange={(e) => setRoleOverride(role, 'model', e.target.value)}
                >
                  {ORCHESTRATOR_MODELS.map((m) => (
                    <option key={m} value={m}>
                      {m.includes('70b') ? '70B · heavy' : '8B · light'}
                    </option>
                  ))}
                </select>
              ) : (
                <span className={`model-tag ${isHeavy ? 'heavy' : 'light'}`}>
                  {isHeavy ? '70B' : '8B'}
                </span>
              )}

              <span className="weight">
                {isManual ? (
                  <input
                    type="range"
                    min={0.2}
                    max={2.5}
                    step={0.1}
                    value={currentWeight}
                    onChange={(e) =>
                      setRoleOverride(role, 'verdict_weight', Number(e.target.value))
                    }
                    className="vm-orch-weight-slider"
                  />
                ) : (
                  bar(d.verdict_weight / 2.0, 60)
                )}
                <span className="wval">{Number(currentWeight).toFixed(1)}×</span>
              </span>

              <span className="attn">
                attn {bar(d.attention_score, 60)} {(d.attention_score * 100).toFixed(0)}%
              </span>
            </div>
          )
        })}
      </div>

      {err && <div className="vm-orch-err">{err}</div>}
      {loading && !plan && (
        <div className="vm-orch-empty">computing plan…</div>
      )}
    </div>
  )
}
