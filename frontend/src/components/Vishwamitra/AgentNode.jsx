import { Handle, Position } from '@xyflow/react'

// Color is provided via data.color (set per-role in SwarmGraph).
// rgb is provided via data.rgb (e.g., "56, 189, 248") so CSS rules
// can mix it into rgba() without parsing the hex at runtime.
//
// data shape:
//   { name, role, color, rgb, action_vector: number[8], confidence: number,
//     personaId, abstain: bool, onHover(payload, evt), onLeave() }
export default function AgentNode({ data }) {
  const {
    name, color, rgb, action_vector = [], confidence = 0,
    abstain = false, tag = '', onHover, onLeave, personaId,
  } = data

  function handleEnter(e) {
    if (onHover) onHover(personaId, e)
  }
  function handleLeave() {
    if (onLeave) onLeave()
  }
  function handleMove(e) {
    if (onHover) onHover(personaId, e)
  }

  const display = (name || '').split(',')[0]
  const conf = Math.round(confidence * 100)
  const isActive = !abstain && confidence > 0.0
  const isHighConf = !abstain && confidence >= 0.6

  // CSS variables consumed by .vm-node, .vm-node-pulse styles.
  const nodeStyle = {
    borderLeftColor: color,
    ['--node-color']: color,
    ['--node-rgb']: rgb || '94, 234, 212',
  }

  return (
    <div
      className={[
        'vm-node',
        abstain ? 'abstain' : '',
        isActive ? 'is-active' : '',
        isHighConf ? 'is-high-conf' : '',
      ].filter(Boolean).join(' ')}
      style={nodeStyle}
      onMouseEnter={handleEnter}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {/* Ambient glow layer behind the card content. CSS handles
          breathing animation; element stays inert (pointer-events:none). */}
      <span className="vm-node-glow" aria-hidden="true" />

      {/* invisible handles so react-flow can still wire edges */}
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />

      <div className="vm-node-head">
        <span className="vm-node-dot" style={{ background: color }} />
        <span className="vm-node-name">{display}</span>
        <span className="vm-node-conf">{conf}</span>
      </div>
      {tag && <div className="vm-node-tag">{tag}</div>}
      <div className="vm-node-bars">
        {action_vector.map((v, i) => (
          <div
            key={i}
            className="vm-node-bar"
            style={{
              height: `${Math.max(2, v * 100)}%`,
              background: v > 0.66 ? color : v > 0.33 ? '#7d8aa3' : '#374055',
            }}
          />
        ))}
      </div>
    </div>
  )
}
