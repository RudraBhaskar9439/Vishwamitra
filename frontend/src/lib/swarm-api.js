// Vishwamitra swarm-deliberation API client.
// Same-origin by default (works under Vite proxy and HF Spaces deployment).
// Override locally with VITE_BACKEND_URL.

const BASE = import.meta.env.VITE_BACKEND_URL ?? ''

export async function getSwarmInfo() {
  const res = await fetch(`${BASE}/swarms/info`)
  if (!res.ok) throw new Error(`/swarms/info → ${res.status}`)
  return res.json()
}

export async function deliberate({ state, scenario, orchestratorOverrides = null }) {
  const body = { state, scenario }
  if (orchestratorOverrides) body.orchestrator_overrides = orchestratorOverrides
  const res = await fetch(`${BASE}/swarms/deliberate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = ''
    try { detail = (await res.json()).detail || '' } catch {}
    throw new Error(`/swarms/deliberate → ${res.status}${detail ? ': ' + detail : ''}`)
  }
  return res.json()
}

export async function previewPlan({ state, orchestratorOverrides = null }) {
  const body = { state }
  if (orchestratorOverrides) body.orchestrator_overrides = orchestratorOverrides
  const res = await fetch(`${BASE}/swarms/preview-plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = ''
    try { detail = (await res.json()).detail || '' } catch {}
    throw new Error(`/swarms/preview-plan → ${res.status}${detail ? ': ' + detail : ''}`)
  }
  return res.json()
}

export const ORCHESTRATOR_ROLES = ['student', 'teacher', 'admin', 'policymaker']
export const ORCHESTRATOR_MODELS = [
  'llama-3.3-70b-versatile',  // heavy
  'llama-3.1-8b-instant',     // light
]

// Generates an IEEE-style narrative policy paper from a prior deliberation
// report. This makes one LLM call (~15-30s) and returns structured prose.
export async function generatePolicyReport({ report, state, scenario }) {
  const res = await fetch(`${BASE}/swarms/policy-report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ report, state, scenario }),
  })
  if (!res.ok) {
    let detail = ''
    try { detail = (await res.json()).detail || '' } catch {}
    throw new Error(`/swarms/policy-report → ${res.status}${detail ? ': ' + detail : ''}`)
  }
  return res.json()
}

// Default state values used by the StateEditor sliders.
// Keys mirror env/state.SystemState fields the swarms understand.
export const DEFAULT_STATE = {
  enrollment_rate: 0.62,
  attendance_rate: 0.55,
  dropout_rate: 0.28,
  teacher_retention: 0.71,
  budget_utilization: 0.92,
  avg_class_size: 48.0,
  teacher_workload: 0.85,
  resource_allocation: 0.40,
  student_engagement: 0.50,
  teacher_burnout: 0.72,
  policy_compliance: 0.65,
  budget_remaining: 420000.0,
  step: 12,
  trust_score: 0.55,
  data_integrity: 0.85,
}

export const DEFAULT_SCENARIO =
  'Funding cut: state slashed the education budget by 35% mid-year. ' +
  'Class sizes ballooned, two teachers resigned last month, dropout signals ' +
  'rising in 9th and 10th grade. Decide intervention intensities for the next quarter.'

// Slider configuration — order = display order. Format hints kept here so the
// editor stays purely presentational.
export const SLIDER_FIELDS = [
  { key: 'enrollment_rate',     label: 'Enrollment',        min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2) },
  { key: 'attendance_rate',     label: 'Attendance',        min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2) },
  { key: 'dropout_rate',        label: 'Dropout rate',      min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2), invert: true },
  { key: 'teacher_retention',   label: 'Teacher retention', min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2) },
  { key: 'teacher_burnout',     label: 'Teacher burnout',   min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2), invert: true },
  { key: 'student_engagement',  label: 'Student engagement',min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2) },
  { key: 'resource_allocation', label: 'Resource allocation',min: 0,  max: 1,    step: 0.01, format: (v) => v.toFixed(2) },
  { key: 'avg_class_size',      label: 'Class size',        min: 15,  max: 60,   step: 1,    format: (v) => v.toFixed(0) },
  { key: 'budget_remaining',    label: 'Budget remaining',  min: 0,   max: 2_000_000, step: 10_000, format: (v) => '$' + (v / 1000).toFixed(0) + 'K' },
  { key: 'trust_score',         label: 'Trust score',       min: 0,   max: 1,    step: 0.01, format: (v) => v.toFixed(2) },
]
