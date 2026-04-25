import { Handle, Position } from '@xyflow/react'

// Color is provided via data.color (set per-role in SwarmGraph).
// data shape:
//   { name, role, color, action_vector: number[8], confidence: number,
//     personaId, abstain: bool, onHover(payload, evt), onLeave() }
export default function AgentNode({ data }) {
  const {
    name, color, action_vector = [], confidence = 0,
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

  return (
    <div
      className={`vm-node ${abstain ? 'abstain' : ''}`}
      style={{ borderLeftColor: color }}
      onMouseEnter={handleEnter}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
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
