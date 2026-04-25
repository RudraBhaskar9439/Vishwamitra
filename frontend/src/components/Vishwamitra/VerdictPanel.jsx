import { useState } from 'react'
import { ROLE_COLORS } from './SwarmGraph.jsx'
import PerSwarmVerdicts from './PerSwarmVerdicts.jsx'
import { renderIEEEPolicyPDF } from '../../lib/policy-pdf.js'
import { generatePolicyReport } from '../../lib/swarm-api.js'

const ROLE_ORDER = ['student', 'teacher', 'admin', 'policymaker']

function ActionRow({ label, value, color, isFlag }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  let cls = 'barfill'
  if (color === 'amber') cls += ' warn'
  if (color === 'bad')   cls += ' bad'
  if (color === 'good')  cls += ' good'
  return (
    <div className="vm-action-row">
      <span className={`label ${isFlag ? 'flag' : ''}`}>{label}</span>
      <span className="barbg"><span className={cls} style={{ width: pct + '%' }} /></span>
      <span className="v">{value.toFixed(2)}</span>
    </div>
  )
}

function colorForResonance(r) {
  if (r > 0.75) return 'good'
  if (r > 0.55) return 'amber'
  return 'bad'
}

export default function VerdictPanel({ report, state, scenario, info }) {
  const [exporting, setExporting] = useState(false)
  const [exportErr, setExportErr] = useState(null)

  if (!report) {
    return (
      <>
        <div className="vm-section-h">Verdict</div>
        <div className="vm-empty">
          <div className="arrow">▾</div>
          set state vector → run deliberation<br />
          <span style={{ color: 'var(--vm-text-mute)', fontSize: 10, letterSpacing: '0.08em' }}>
            VERDICT WILL APPEAR HERE
          </span>
        </div>
      </>
    )
  }

  const names = report.action_names || []
  const final = report.final_action || []
  const reson = report.resonance_per_intervention || []
  const flags = new Set(report.dissonance_flags || [])
  const swarmsByRole = Object.fromEntries((report.swarm_verdicts || []).map((s) => [s.role, s]))

  async function handleDownload() {
    if (exporting) return
    setExporting(true); setExportErr(null)
    try {
      const paper = await generatePolicyReport({ report, state, scenario })
      renderIEEEPolicyPDF({ paper, report })
    } catch (e) {
      console.error('IEEE report generation failed:', e)
      setExportErr(String(e.message || e))
    } finally {
      setExporting(false)
    }
  }

  return (
    <>
      <div className="vm-section-h">Verdict</div>
      <div className="vm-verdict-scroll">

        <div className="vm-block">
          <div className="vm-block-title">Final Action Vector</div>
          {names.map((n, i) => (
            <ActionRow
              key={n}
              label={n}
              value={final[i] ?? 0}
              color={null}
              isFlag={flags.has(n)}
            />
          ))}
        </div>

        <div className="vm-block">
          <div className="vm-block-title">Resonance per Intervention</div>
          {names.map((n, i) => (
            <ActionRow
              key={n}
              label={n}
              value={reson[i] ?? 0}
              color={colorForResonance(reson[i] ?? 0)}
              isFlag={flags.has(n)}
            />
          ))}
        </div>

        <div className="vm-block">
          <div className="vm-block-title">Dissonance Flags</div>
          {flags.size === 0 ? (
            <div className="vm-no-flags">● ALL SWARMS RESONATED</div>
          ) : (
            <div className="vm-flags">
              {[...flags].map((f) => <span key={f} className="vm-flag">{f}</span>)}
            </div>
          )}
        </div>

        <PerSwarmVerdicts report={report} />

        <div className="vm-block">
          <div className="vm-block-title">Per-Role Aggregated</div>
          <div className="vm-perrole">
            {ROLE_ORDER.map((role) => {
              const sv = swarmsByRole[role]
              const color = ROLE_COLORS[role]
              const vec = (sv && sv.aggregated_action) || []
              const conf = sv ? sv.mean_confidence : 0
              return (
                <div key={role} className="vm-perrole-card">
                  <div className="ttl" style={{ color }}>
                    {role} · {(conf * 100).toFixed(0)}%
                  </div>
                  <div className="vm-perrole-bars">
                    {vec.map((v, i) => (
                      <div
                        key={i}
                        className="vm-perrole-bar"
                        style={{
                          height: `${Math.max(2, v * 100)}%`,
                          background: v > 0.5 ? color : '#374055',
                        }}
                      />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

      </div>

      <div className="vm-export-row">
        <button
          className={`vm-export-btn ${exporting ? 'loading' : ''}`}
          onClick={handleDownload}
          disabled={exporting}
        >
          {exporting
            ? '◐ Synthesising IEEE Report…'
            : '⬇ Generate IEEE Policy Paper (PDF)'}
        </button>
        {exportErr ? (
          <div className="vm-export-sub" style={{ color: 'var(--vm-red)' }}>
            {exportErr}
          </div>
        ) : (
          <div className="vm-export-sub">
            {exporting
              ? 'narrative analysis · projections · ~15–30s'
              : `${(report.swarm_verdicts || []).reduce((acc, s) => acc + (s.verdicts || []).length, 0)} agents · ${flags.size} dissonance flag${flags.size === 1 ? '' : 's'} · LLM-authored`}
          </div>
        )}
      </div>
    </>
  )
}
