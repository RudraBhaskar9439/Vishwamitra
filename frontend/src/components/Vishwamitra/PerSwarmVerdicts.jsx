import { ROLE_COLORS } from './SwarmGraph.jsx'

const ROLE_ORDER = ['student', 'teacher', 'admin', 'policymaker']

const ACTION_NAMES = [
  'funding_boost','teacher_incentive','student_scholarship','attendance_mandate',
  'resource_realloc','transparency_report','staff_hiring','counseling_programs',
]

function FitBadge({ fit }) {
  if (!fit) return null
  const w = Number(fit.weight ?? 1).toFixed(2)
  const fitPct = Math.round((fit.fit_score ?? 0) * 100)
  // tag tier: high (>1.2), neutral (0.7..1.2), low (<0.7)
  const tier = (fit.weight ?? 1) > 1.2 ? 'high' : (fit.weight ?? 1) < 0.7 ? 'low' : 'mid'
  const matched = fit.matched_signals || {}
  const topSignals = Object.entries(matched)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([k]) => k)
    .join(', ')
  return (
    <span
      className={`vm-fit-badge ${tier}`}
      title={`L2 fit weight ${w}× · fit_score ${fitPct}%${topSignals ? '\nTop matched signals: ' + topSignals : ''}`}
    >
      L2 {w}×
    </span>
  )
}

function AgentCard({ verdict, color, fit }) {
  if (!verdict) return null
  const conf = Math.round(verdict.confidence * 100)
  const ranked = (verdict.action_vector || [])
    .map((v, i) => ({ name: ACTION_NAMES[i], v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 3)
  return (
    <div className={`vm-agent-card ${verdict.error ? 'abstain' : ''}`}>
      <div className="vm-agent-head">
        <span className="dot" style={{ background: color }} />
        <span className="name">{verdict.persona_name}</span>
        <FitBadge fit={fit} />
        <span className="conf">{conf}%</span>
      </div>
      {fit && Object.keys(fit.matched_signals || {}).length > 0 && (
        <div className="vm-fit-signals">
          {Object.entries(fit.matched_signals)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4)
            .map(([k, v]) => (
              <span key={k} className="vm-fit-chip">
                {k} <b>{Number(v).toFixed(2)}</b>
              </span>
            ))}
        </div>
      )}
      {verdict.error ? (
        <div className="vm-agent-reason muted">
          (abstained — {verdict.error.slice(0, 80)})
        </div>
      ) : (
        <>
          <div className="vm-agent-reason">"{verdict.reasoning}"</div>
          <div className="vm-agent-bars">
            {(verdict.action_vector || []).map((v, i) => (
              <div
                key={i}
                className="vm-agent-bar"
                title={`${ACTION_NAMES[i]}: ${v.toFixed(2)}`}
                style={{
                  height: `${Math.max(2, v * 100)}%`,
                  background: v > 0.5 ? color : '#374055',
                }}
              />
            ))}
          </div>
          <div className="vm-agent-top">
            {ranked.map((r) => (
              <div key={r.name} className="vm-agent-top-row">
                <span className="n">{r.name}</span>
                <span className="bar"><span className="fill" style={{ width: `${r.v * 100}%`, background: color }} /></span>
                <span className="v">{r.v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export default function PerSwarmVerdicts({ report, defaultOpen = null }) {
  if (!report) return null
  const swarmsByRole = Object.fromEntries((report.swarm_verdicts || []).map((s) => [s.role, s]))

  return (
    <div className="vm-block">
      <div className="vm-block-title">Per-Swarm Verdicts</div>
      <div className="vm-acc-list">
        {ROLE_ORDER.map((role) => {
          const sv = swarmsByRole[role]
          if (!sv) return null
          const color = ROLE_COLORS[role]
          const conf = Math.round(sv.mean_confidence * 100)
          // L2 fit decisions per persona, keyed by persona_id
          const fitById = Object.fromEntries(
            (sv.persona_fits || []).map((f) => [f.persona_id, f])
          )
          // Highlight the dominant persona (highest L2 weight in this swarm)
          const dominant = (sv.persona_fits || [])
            .slice()
            .sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0))[0]
          return (
            <details
              key={role}
              className="vm-acc"
              open={defaultOpen === role}
            >
              <summary className="vm-acc-head" style={{ borderLeftColor: color }}>
                <span className="dot" style={{ background: color }} />
                <span className="name">{role}</span>
                <span className="meta">
                  {sv.verdicts.length} agents · {conf}%
                  {dominant && (
                    <> · L2 leader: <b style={{ color }}>{dominant.persona_name?.split(',')[0]}</b></>
                  )}
                </span>
                <span className="chev">▾</span>
              </summary>
              <div className="vm-acc-body">
                {sv.verdicts.map((v) => (
                  <AgentCard
                    key={v.persona_id}
                    verdict={v}
                    color={color}
                    fit={fitById[v.persona_id]}
                  />
                ))}
              </div>
            </details>
          )
        })}
      </div>
    </div>
  )
}
