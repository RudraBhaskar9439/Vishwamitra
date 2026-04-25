import { ROLE_COLORS } from './SwarmGraph.jsx'

const ROLE_ORDER = ['student', 'teacher', 'admin', 'policymaker']

const ACTION_NAMES = [
  'funding_boost','teacher_incentive','student_scholarship','attendance_mandate',
  'resource_realloc','transparency_report','staff_hiring','counseling_programs',
]

function AgentCard({ verdict, color }) {
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
        <span className="conf">{conf}%</span>
      </div>
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
          return (
            <details
              key={role}
              className="vm-acc"
              open={defaultOpen === role}
            >
              <summary className="vm-acc-head" style={{ borderLeftColor: color }}>
                <span className="dot" style={{ background: color }} />
                <span className="name">{role}</span>
                <span className="meta">{sv.verdicts.length} agents · {conf}%</span>
                <span className="chev">▾</span>
              </summary>
              <div className="vm-acc-body">
                {sv.verdicts.map((v) => (
                  <AgentCard key={v.persona_id} verdict={v} color={color} />
                ))}
              </div>
            </details>
          )
        })}
      </div>
    </div>
  )
}
