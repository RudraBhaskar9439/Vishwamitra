import { useEffect, useState } from 'react'

const ACTION_NAMES = [
  'funding_boost','teacher_incentive','student_scholarship','attendance_mandate',
  'resource_realloc','transparency_report','staff_hiring','counseling_programs',
]

export default function AgentTooltip({ verdict, role, color, mouseRef }) {
  // Re-render on mouse move (mouseRef is updated by parent on mouseMove).
  const [, setTick] = useState(0)
  useEffect(() => {
    let raf
    const loop = () => { setTick((t) => t + 1); raf = requestAnimationFrame(loop) }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  if (!verdict) return null
  const { x, y } = mouseRef.current || { x: 0, y: 0 }

  // Place to the right of cursor unless near right edge.
  const W = 360
  const margin = 16
  const left = x + 18 + W + margin > window.innerWidth ? x - W - 18 : x + 18
  const top  = Math.min(window.innerHeight - 280, y + 12)

  // Top-3 highest action values in this verdict.
  const ranked = (verdict.action_vector || [])
    .map((v, i) => ({ name: ACTION_NAMES[i], v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 3)

  return (
    <div className="vm-tooltip" style={{ left, top }}>
      <div className="vm-tooltip-head">
        <span className="vm-tooltip-name">{verdict.persona_name}</span>
        <span className="vm-tooltip-role" style={{ color }}>{role}</span>
      </div>
      {verdict.reasoning ? (
        <div className="vm-tooltip-reason">{verdict.reasoning}</div>
      ) : (
        <div className="vm-tooltip-reason" style={{ color: 'var(--vm-text-mute)' }}>
          (no reasoning returned)
        </div>
      )}
      <div className="vm-tooltip-meta">
        <span>conf <b>{(verdict.confidence * 100).toFixed(0)}%</b></span>
        <span>id <b>{verdict.persona_id}</b></span>
        {verdict.error && <span style={{ color: 'var(--vm-red)' }}>err</span>}
      </div>
      <div className="vm-tooltip-actions">
        <div style={{ color: 'var(--vm-text-mute)', marginBottom: 4 }}>TOP RECOMMENDATIONS</div>
        {ranked.map((r) => (
          <div key={r.name} className="vm-tooltip-action">
            <span style={{ width: 130, color: 'var(--vm-text-dim)' }}>{r.name}</span>
            <span className="bar"><span className="fill" style={{ width: `${r.v * 100}%`, background: color }} /></span>
            <span className="v">{r.v.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
