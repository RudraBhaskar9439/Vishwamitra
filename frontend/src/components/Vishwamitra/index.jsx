import { useEffect, useMemo, useRef, useState } from 'react'
import StateEditor from './StateEditor.jsx'
import SwarmGraph, { ROLE_COLORS } from './SwarmGraph.jsx'
import VerdictPanel from './VerdictPanel.jsx'
import AgentTooltip from './AgentTooltip.jsx'
import OrchestratorPanel from './OrchestratorPanel.jsx'
import { deliberate, getSwarmInfo, DEFAULT_STATE, DEFAULT_SCENARIO } from '../../lib/swarm-api.js'
import './styles.css'

export default function Vishwamitra({ onBack }) {
  const [state, setState] = useState(DEFAULT_STATE)
  const [scenario, setScenario] = useState(DEFAULT_SCENARIO)
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [info, setInfo] = useState(null)
  const [hoveredId, setHoveredId] = useState(null)
  const [orchMode, setOrchMode] = useState('auto')         // 'auto' | 'manual'
  const [orchOverrides, setOrchOverrides] = useState({})   // { role: { model, verdict_weight } }

  const mouseRef = useRef({ x: 0, y: 0 })

  // pull provider/model info on mount (best-effort)
  useEffect(() => {
    getSwarmInfo()
      .then(setInfo)
      .catch(() => setInfo(null))
  }, [])

  // page-wide mouse tracker for the floating tooltip
  useEffect(() => {
    function track(e) { mouseRef.current = { x: e.clientX, y: e.clientY } }
    window.addEventListener('mousemove', track)
    return () => window.removeEventListener('mousemove', track)
  }, [])

  async function runDeliberation() {
    setLoading(true); setError(null)
    const orchestratorOverrides =
      orchMode === 'manual' && Object.keys(orchOverrides).length
        ? { mode: 'manual', roles: orchOverrides }
        : null
    try {
      const r = await deliberate({ state, scenario, orchestratorOverrides })
      setReport(r)
    } catch (e) {
      setError(String(e.message || e))
      setReport(null)
    } finally {
      setLoading(false)
    }
  }

  // O(1) lookup: persona_id → { verdict, role, color }
  const verdictIndex = useMemo(() => {
    const map = new Map()
    if (!report) return map
    for (const sv of report.swarm_verdicts || []) {
      for (const v of sv.verdicts || []) {
        map.set(v.persona_id, { verdict: v, role: sv.role, color: ROLE_COLORS[sv.role] })
      }
    }
    return map
  }, [report])

  function handleAgentHover(personaId, evt) {
    setHoveredId(personaId)
    if (evt) mouseRef.current = { x: evt.clientX, y: evt.clientY }
  }
  function handleAgentLeave() { setHoveredId(null) }

  const hovered = hoveredId ? verdictIndex.get(hoveredId) : null

  // Topbar status indicator
  const statusTone = loading ? 'amber' : error ? 'red' : 'green'
  const statusText = loading ? 'Deliberating' : error ? 'Error' : (report ? 'Ready' : 'Idle')

  // Bottom statusbar
  const lastTs = report ? new Date(report.timestamp).toLocaleTimeString() : '—'
  const verdictsCount = report
    ? (report.swarm_verdicts || []).reduce((acc, s) => acc + (s.verdicts || []).length, 0)
    : 0

  return (
    <div className="vm-page">

      {/* TOP BAR */}
      <div className="vm-topbar">
        <span className="vm-topbar-brand">VISHWAMITRA</span>
        <span className="vm-topbar-sub">SWARM DELIBERATION</span>

        <span className="vm-topbar-spacer" />

        {info && (
          <>
            <span className="vm-topbar-pill">model · {info.model || '—'}</span>
            <span className="vm-topbar-pill">provider · {info.provider || '—'}</span>
            <span className="vm-topbar-pill">
              roles · {Object.keys(info.roles || {}).length}
            </span>
          </>
        )}
        <span className="vm-topbar-pill">
          <span className={`vm-status-dot ${statusTone === 'green' ? '' : statusTone}`} />
          {statusText}
        </span>
        {onBack && <button className="vm-topbar-back" onClick={onBack}>← Back</button>}
      </div>

      {/* MAIN GRID */}
      <div className="vm-grid">

        <div className="vm-pane left">
          <StateEditor
            state={state}
            scenario={scenario}
            onStateChange={setState}
            onScenarioChange={setScenario}
            onRun={runDeliberation}
            loading={loading}
            error={error}
          />
        </div>

        <div className="vm-pane center">
          <OrchestratorPanel
            state={state}
            mode={orchMode}
            onModeChange={setOrchMode}
            overrides={orchOverrides}
            onOverridesChange={setOrchOverrides}
          />
          <div className="vm-section-h">Swarm Topology</div>
          <SwarmGraph
            report={report}
            onAgentHover={handleAgentHover}
            onAgentLeave={handleAgentLeave}
          />
        </div>

        <div className="vm-pane right">
          <VerdictPanel report={report} state={state} scenario={scenario} info={info} />
        </div>

      </div>

      {/* BOTTOM STATUS */}
      <div className="vm-statusbar">
        <span>VISHWAMITRA · v0.1</span>
        <span className="sep">│</span>
        <span>last run: {lastTs}</span>
        <span className="sep">│</span>
        <span>verdicts: {verdictsCount}</span>
        <span className="sep">│</span>
        <span>scenario: {(report?.scenario || scenario || '').slice(0, 60)}{(report?.scenario || scenario || '').length > 60 ? '…' : ''}</span>
      </div>

      {/* HOVER TOOLTIP */}
      {hovered && (
        <AgentTooltip
          verdict={hovered.verdict}
          role={hovered.role}
          color={hovered.color}
          mouseRef={mouseRef}
        />
      )}
    </div>
  )
}
