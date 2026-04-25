import { useMemo } from 'react'
import { ReactFlow, Background, Controls, BackgroundVariant } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import AgentNode from './AgentNode.jsx'

export const ROLE_COLORS = {
  student:     '#38bdf8',
  teacher:     '#fbbf24',
  admin:       '#c084fc',
  policymaker: '#4ade80',
}

const ROLE_ORDER = ['student', 'teacher', 'admin', 'policymaker']

const CLUSTER_W = 360
const CLUSTER_H = 240
const CLUSTER_GAP_X = 80
const CLUSTER_GAP_Y = 80

const CLUSTER_POS = {
  student:     { x: 0,                              y: 0 },
  teacher:     { x: CLUSTER_W + CLUSTER_GAP_X,      y: 0 },
  admin:       { x: 0,                              y: CLUSTER_H + CLUSTER_GAP_Y },
  policymaker: { x: CLUSTER_W + CLUSTER_GAP_X,      y: CLUSTER_H + CLUSTER_GAP_Y },
}

// Three slots per cluster, triangle layout (in cluster-local coords).
const SLOT_OFFSETS = [
  { x: 100, y: 36 },   // top
  { x: 24,  y: 142 },  // bottom-left
  { x: 188, y: 142 },  // bottom-right
]

const nodeTypes = { agent: AgentNode }

// Build a placeholder shell so the graph renders even before deliberation.
function buildPlaceholder() {
  const nodes = []
  ROLE_ORDER.forEach((role) => {
    const pos = CLUSTER_POS[role]
    nodes.push({
      id: `cluster_${role}`,
      type: 'group',
      position: pos,
      data: { label: role },
      style: {
        width: CLUSTER_W,
        height: CLUSTER_H,
        background: 'rgba(255,255,255,0.015)',
        border: `1px dashed ${ROLE_COLORS[role]}55`,
        borderRadius: 6,
      },
      draggable: false,
      selectable: false,
    })
    SLOT_OFFSETS.forEach((slot, i) => {
      nodes.push({
        id: `${role}_slot_${i}`,
        type: 'agent',
        parentId: `cluster_${role}`,
        extent: 'parent',
        position: slot,
        data: {
          name: `agent ${i + 1}`,
          role,
          color: ROLE_COLORS[role],
          action_vector: [0, 0, 0, 0, 0, 0, 0, 0],
          confidence: 0,
          abstain: true,
          tag: 'awaiting deliberation',
          personaId: null,
        },
        draggable: false,
      })
    })
  })
  return { nodes, edges: [] }
}

function buildFromReport(report, onHover, onLeave) {
  const nodes = []
  const edges = []
  if (!report) return { nodes, edges }

  const swarmsByRole = Object.fromEntries(
    (report.swarm_verdicts || []).map((sv) => [sv.role, sv])
  )

  ROLE_ORDER.forEach((role) => {
    const sv = swarmsByRole[role]
    const pos = CLUSTER_POS[role]
    const conf = sv ? sv.mean_confidence : 0
    const color = ROLE_COLORS[role]

    nodes.push({
      id: `cluster_${role}`,
      type: 'group',
      position: pos,
      data: {},
      style: {
        width: CLUSTER_W,
        height: CLUSTER_H,
        background: `radial-gradient(ellipse at center, ${color}0a, transparent 70%)`,
        border: `1px dashed ${color}66`,
        borderRadius: 6,
      },
      draggable: false,
      selectable: false,
    })

    const verdicts = (sv && sv.verdicts) || []
    const triEdgeFrom = []

    verdicts.slice(0, 3).forEach((v, i) => {
      const slot = SLOT_OFFSETS[i]
      const nodeId = `${role}__${v.persona_id}`
      triEdgeFrom.push(nodeId)
      nodes.push({
        id: nodeId,
        type: 'agent',
        parentId: `cluster_${role}`,
        extent: 'parent',
        position: slot,
        data: {
          name: v.persona_name,
          role,
          color,
          action_vector: v.action_vector,
          confidence: v.confidence,
          abstain: v.error != null,
          tag: '',
          personaId: v.persona_id,
          onHover, onLeave,
        },
        draggable: false,
      })
    })

    // Triangle edges within cluster (faint).
    if (triEdgeFrom.length === 3) {
      const [a, b, c] = triEdgeFrom
      edges.push(
        { id: `${role}-ab`, source: a, target: b, style: { stroke: `${color}55`, strokeWidth: 1 } },
        { id: `${role}-bc`, source: b, target: c, style: { stroke: `${color}55`, strokeWidth: 1 } },
        { id: `${role}-ca`, source: c, target: a, style: { stroke: `${color}55`, strokeWidth: 1 } },
      )
    }

    // Confidence bar overlay handled via node label later; skip for now.
    void conf
  })

  // Cross-cluster edges: connect cluster centroids using the FIRST agent of each cluster.
  // Color reflects mean resonance; thinner if low resonance (more dissent).
  const meanResonance = (() => {
    const arr = (report.resonance_per_intervention || []).filter((x) => Number.isFinite(x))
    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 1
  })()
  const edgeColor = meanResonance > 0.75
    ? '#4ade8077'
    : meanResonance > 0.55
    ? '#fbbf2477'
    : '#f8717177'
  const firstAgents = ROLE_ORDER
    .map((role) => {
      const sv = swarmsByRole[role]
      const v = sv && sv.verdicts && sv.verdicts[0]
      return v ? `${role}__${v.persona_id}` : null
    })
    .filter(Boolean)

  for (let i = 0; i < firstAgents.length; i++) {
    for (let j = i + 1; j < firstAgents.length; j++) {
      edges.push({
        id: `cross-${i}-${j}`,
        source: firstAgents[i],
        target: firstAgents[j],
        style: { stroke: edgeColor, strokeWidth: 1, strokeDasharray: '4 4' },
        animated: false,
      })
    }
  }

  return { nodes, edges }
}

export default function SwarmGraph({ report, onAgentHover, onAgentLeave }) {
  const { nodes, edges } = useMemo(() => {
    if (!report) return buildPlaceholder()
    return buildFromReport(report, onAgentHover, onAgentLeave)
  }, [report, onAgentHover, onAgentLeave])

  return (
    <div className="vm-graph-wrap">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: false }}
      >
        <Background variant={BackgroundVariant.Dots} gap={28} size={1} color="#1a2030" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
