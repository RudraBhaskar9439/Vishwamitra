// In-depth Educational Policy Brief PDF renderer.
//
// Renders the structured prose returned by /swarms/policy-report — which
// is now produced via two sequential LLM calls — into a multi-section
// professional document.
//
// Document layout:
//   1. Title page (brand banner, title, subtitle, executive summary box)
//   2. What is Educational Policy?
//   3. Scenario Analysis
//        - Operational Context (paragraphs)
//        - State Diagnostic (3-col table)
//        - Root Cause Hypothesis
//        - Stakes of Inaction
//        - Success Criteria (bullets)
//   4. The Policymaking Process — Key Stages (6 stages with bullets)
//   5. Implementation Roadmap (4 phase blocks)
//   6. Risk Register (table with likelihood / impact tiers)
//   7. Key Stakeholders (table)
//   8. Stakeholder Feedback Synthesis (per-persona cards with quotes)
//   9. Areas of Agreement vs. Contention
//  10. The Iterative Nature of the Process
//  11. Challenges in Educational Policymaking
//  12. Strategies for Effective Implementation
//  13. Key Takeaway
//  14. Appendix · Source Deliberation Snapshot

import jsPDF from 'jspdf'

// Backwards-compat alias
export function renderIEEEPolicyPDF({ paper, report }) {
  return renderPolicyBriefPDF({ paper, report })
}

const ACTION_NAMES = [
  'funding_boost','teacher_incentive','student_scholarship','attendance_mandate',
  'resource_realloc','transparency_report','staff_hiring','counseling_programs',
]

const TIER_COLORS = {
  low:    [74, 140, 90],    // green
  medium: [200, 140, 30],   // amber
  high:   [190, 50, 50],    // red
}

export function renderPolicyBriefPDF({ paper, report }) {
  const pdf = new jsPDF({ unit: 'pt', format: 'letter' })
  const PW = pdf.internal.pageSize.getWidth()
  const PH = pdf.internal.pageSize.getHeight()
  const M = 60
  const W = PW - 2 * M

  let y = M
  let pageNum = 1
  const runningTitle = (paper.title || 'Educational Policy Brief').slice(0, 90)

  // ------------ helpers ------------
  function drawFooter() {
    pdf.setFont('times', 'italic')
    pdf.setFontSize(8.5)
    pdf.setTextColor(110)
    pdf.text(runningTitle, M, PH - 28)
    pdf.text(`${pageNum}`, PW - M, PH - 28, { align: 'right' })
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(180)
    pdf.line(M, PH - 36, PW - M, PH - 36)
  }
  function newPage() {
    drawFooter()
    pdf.addPage()
    pageNum += 1
    y = M
  }
  function ensureSpace(needed) {
    if (y + needed > PH - M - 36) newPage()
  }
  function hrThin(weight = 0.4, color = 200) {
    pdf.setLineWidth(weight)
    pdf.setDrawColor(color)
    pdf.line(M, y, PW - M, y)
    y += 8
  }
  function h1Centered(text, size = 22, color = [15, 23, 42]) {
    pdf.setFont('times', 'bold')
    pdf.setFontSize(size)
    pdf.setTextColor(...color)
    const lines = pdf.splitTextToSize(text, W)
    for (const line of lines) {
      pdf.text(line, PW / 2, y, { align: 'center' })
      y += size + 4
    }
  }
  function h2(text, accent = [60, 80, 200]) {
    ensureSpace(40)
    y += 4
    pdf.setFillColor(...accent)
    pdf.rect(M, y - 9, 4, 16, 'F')
    pdf.setFont('times', 'bold')
    pdf.setFontSize(15)
    pdf.setTextColor(15, 23, 42)
    pdf.text(text, M + 12, y)
    y += 10
    hrThin(0.5, 150)
    y += 4
  }
  function h3(text) {
    ensureSpace(28)
    pdf.setFont('times', 'bold')
    pdf.setFontSize(12)
    pdf.setTextColor(20, 30, 50)
    pdf.text(text, M, y)
    y += 14
  }
  function paragraph(text, opts = {}) {
    if (!text || !text.trim()) return
    pdf.setFont('times', opts.italic ? 'italic' : 'normal')
    pdf.setFontSize(opts.size || 10.5)
    pdf.setTextColor(...(opts.color || [30, 35, 50]))
    const paras = text.split(/\n\s*\n+/).map((p) => p.trim().replace(/\s+/g, ' ')).filter(Boolean)
    const lh = (opts.size || 10.5) * 1.42
    const indentX = opts.indent || 0
    for (const para of paras) {
      const lines = pdf.splitTextToSize(para, W - indentX)
      ensureSpace(lines.length * lh + 4)
      lines.forEach((ln) => {
        pdf.text(ln, M + indentX, y)
        y += lh
      })
      y += 4
    }
  }
  function bullets(items, { indent = 14, size = 10.5 } = {}) {
    if (!items || !items.length) return
    pdf.setFontSize(size)
    pdf.setTextColor(30, 35, 50)
    const lh = size * 1.42
    const textWidth = W - indent
    for (const raw of items) {
      const item = String(raw || '').trim().replace(/\s+/g, ' ')
      if (!item) continue
      pdf.setFont('times', 'normal')
      const lines = pdf.splitTextToSize(item, textWidth)
      ensureSpace(lines.length * lh + 4)
      pdf.setFont('times', 'bold')
      pdf.text('•', M, y)
      pdf.setFont('times', 'normal')
      lines.forEach((ln) => {
        pdf.text(ln, M + indent, y)
        y += lh
      })
      y += 2
    }
    y += 2
  }
  function inlineMeta(label, value) {
    if (!value || !value.trim()) return
    ensureSpace(22)
    pdf.setFont('times', 'italic')
    pdf.setFontSize(10)
    pdf.setTextColor(70, 80, 100)
    const labelStr = label + ': '
    pdf.text(labelStr, M + 10, y)
    const offset = pdf.getTextWidth(labelStr)
    pdf.setFont('times', 'normal')
    pdf.setTextColor(40, 45, 60)
    const lines = pdf.splitTextToSize(value.trim(), W - 10 - offset)
    pdf.text(lines[0] || '', M + 10 + offset, y)
    y += 14
    for (let i = 1; i < lines.length; i++) {
      pdf.text(lines[i], M + 10 + offset, y)
      y += 14
    }
    y += 4
  }
  function calloutBox(text, { titleStr = '', accent = [40, 80, 160], padding = 12 } = {}) {
    if (!text || !text.trim()) return
    pdf.setFont('times', 'normal')
    pdf.setFontSize(10.5)
    const lh = 14
    const lines = pdf.splitTextToSize(text.trim(), W - 2 * padding)
    const titleH = titleStr ? 16 : 0
    const total = titleH + lines.length * lh + 2 * padding
    ensureSpace(total + 8)
    pdf.setFillColor(248, 250, 254)
    pdf.setDrawColor(...accent)
    pdf.setLineWidth(0.6)
    pdf.rect(M, y, W, total, 'FD')
    // 3px accent stripe
    pdf.setFillColor(...accent)
    pdf.rect(M, y, 3, total, 'F')
    let yy = y + padding
    if (titleStr) {
      pdf.setFont('times', 'bold')
      pdf.setFontSize(10)
      pdf.setTextColor(...accent)
      pdf.text(titleStr.toUpperCase(), M + padding, yy + 2)
      yy += titleH
    }
    pdf.setFont('times', 'normal')
    pdf.setFontSize(10.5)
    pdf.setTextColor(30, 35, 50)
    lines.forEach((ln) => {
      pdf.text(ln, M + padding, yy + 2)
      yy += lh
    })
    y += total + 10
  }

  // ============================================================
  // PAGE 1 — TITLE
  // ============================================================
  pdf.setFillColor(7, 11, 24)
  pdf.rect(0, 0, PW, 4, 'F')

  y = 92
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(8)
  pdf.setTextColor(94, 234, 212)
  pdf.text('VISHWAMITRA · POLICY DELIBERATION', PW / 2, y, { align: 'center' })
  y += 36

  h1Centered(paper.title || 'Educational Policy Brief')

  if (paper.subtitle && paper.subtitle.trim()) {
    y += 4
    pdf.setFont('times', 'italic')
    pdf.setFontSize(12)
    pdf.setTextColor(60, 75, 95)
    const sub = pdf.splitTextToSize(paper.subtitle.trim(), W - 40)
    sub.forEach((line) => {
      pdf.text(line, PW / 2, y, { align: 'center' })
      y += 16
    })
  }

  y += 6
  pdf.setFont('times', 'italic')
  pdf.setFontSize(10)
  pdf.setTextColor(80, 90, 110)
  pdf.text(
    `Generated ${new Date(report.timestamp).toLocaleString()}`,
    PW / 2, y, { align: 'center' },
  )
  y += 20
  pdf.setLineWidth(0.6)
  pdf.setDrawColor(120)
  pdf.line(M + 80, y, PW - M - 80, y)
  y += 22

  if (paper.executive_summary && paper.executive_summary.trim()) {
    calloutBox(paper.executive_summary, {
      titleStr: 'Executive Summary',
      accent: [40, 80, 160],
    })
  }

  // ============================================================
  // WHAT IS
  // ============================================================
  if (paper.what_is && paper.what_is.trim()) {
    h2('What is Educational Policy?', [40, 80, 160])
    paragraph(paper.what_is)
  }

  // ============================================================
  // SCENARIO ANALYSIS
  // ============================================================
  const hasScenario =
    paper.operational_context || (paper.state_diagnostic || []).length ||
    paper.root_cause_hypothesis || paper.stakes_of_inaction ||
    (paper.success_criteria || []).length

  if (hasScenario) {
    h2('Scenario Analysis', [180, 80, 30])

    if (paper.operational_context) {
      h3('Operational Context')
      paragraph(paper.operational_context)
    }

    if ((paper.state_diagnostic || []).length) {
      h3('State Diagnostic')
      diagnosticTable(paper.state_diagnostic)
    }

    if (paper.root_cause_hypothesis) {
      h3('Root-Cause Hypothesis')
      paragraph(paper.root_cause_hypothesis)
    }

    if (paper.stakes_of_inaction) {
      h3('Stakes of Inaction')
      calloutBox(paper.stakes_of_inaction, {
        titleStr: 'If no action is taken',
        accent: [190, 50, 50],
      })
    }

    if ((paper.success_criteria || []).length) {
      h3('Success Criteria')
      bullets(paper.success_criteria)
    }
  }

  // ============================================================
  // SIX-STAGE PROCESS
  // ============================================================
  h2('The Policymaking Process — Key Stages', [60, 80, 200])

  const stageDefs = [
    { num: 1, name: 'Issue / Problem Identification',
      desc: paper.stage_1_description, bullets: paper.stage_1_bullets },
    { num: 2, name: 'Agenda Setting',
      desc: paper.stage_2_description, bullets: paper.stage_2_bullets,
      meta: [['Key influencers', paper.stage_2_influencers]] },
    { num: 3, name: 'Policy Formulation',
      desc: paper.stage_3_description, bullets: paper.stage_3_bullets,
      meta: [['Contributors', paper.stage_3_contributors]] },
    { num: 4, name: 'Policy Adoption',
      desc: paper.stage_4_description, bullets: paper.stage_4_bullets },
    { num: 5, name: 'Policy Implementation',
      desc: paper.stage_5_description, bullets: paper.stage_5_bullets,
      meta: [['Implementation challenges', paper.stage_5_challenges]] },
    { num: 6, name: 'Policy Evaluation',
      desc: paper.stage_6_description, bullets: paper.stage_6_bullets },
  ]
  for (const s of stageDefs) {
    if (!s.desc && !(s.bullets || []).length) continue
    h3(`Stage ${s.num}: ${s.name}`)
    if (s.desc) paragraph(s.desc)
    bullets(s.bullets)
    if (s.meta) {
      for (const [label, value] of s.meta) inlineMeta(label, value)
    }
    y += 4
  }

  // ============================================================
  // IMPLEMENTATION ROADMAP
  // ============================================================
  if ((paper.roadmap || []).length) {
    h2('Implementation Roadmap', [40, 140, 90])
    paragraph(
      'A four-phase rollout sequencing the recommended interventions. '
      + 'Each phase names its objective, owner roles, milestones, and the '
      + 'dependencies that must be in place before the phase begins.',
      { italic: true, color: [90, 100, 120], size: 9.5 },
    )
    y += 4
    for (const phase of paper.roadmap) {
      drawRoadmapPhase(phase)
    }
  }

  // ============================================================
  // RISK REGISTER
  // ============================================================
  if ((paper.risk_register || []).length) {
    h2('Risk Register', [190, 50, 50])
    paragraph(
      'Risks identified during deliberation, scored on likelihood × impact, '
      + 'with proposed mitigations. At least one risk per dissonance flag.',
      { italic: true, color: [90, 100, 120], size: 9.5 },
    )
    y += 4
    drawRiskTable(paper.risk_register)
  }

  // ============================================================
  // KEY STAKEHOLDERS
  // ============================================================
  if ((paper.stakeholders || []).length) {
    h2('Key Stakeholders', [60, 80, 200])
    const rows = paper.stakeholders.map((s) => [s.name || '', s.role || ''])
    table(['Stakeholder', 'Role in this deliberation'], rows)
  }

  // ============================================================
  // STAKEHOLDER FEEDBACK SYNTHESIS
  // ============================================================
  if ((paper.persona_feedback || []).length) {
    h2('Stakeholder Feedback Synthesis', [120, 80, 180])
    paragraph(
      'Synthesised feedback from each persona that materially shaped the '
      + 'recommendation, organised by what they said, what they want, and '
      + 'what they fear.',
      { italic: true, color: [90, 100, 120], size: 9.5 },
    )
    y += 4
    for (const fb of paper.persona_feedback) {
      drawFeedbackCard(fb)
    }
  }

  // ============================================================
  // CONSENSUS / CONTENTION
  // ============================================================
  if ((paper.areas_of_agreement || []).length || (paper.areas_of_contention || []).length) {
    h2('Areas of Agreement vs. Contention', [40, 80, 160])
    if ((paper.areas_of_agreement || []).length) {
      h3('Where the lenses converged')
      bullets(paper.areas_of_agreement)
    }
    if ((paper.areas_of_contention || []).length) {
      h3('Where the lenses diverged')
      bullets(paper.areas_of_contention)
    }
  }

  // ============================================================
  // ITERATIVE NATURE
  // ============================================================
  if ((paper.iterative_nature || []).length) {
    h2('The Iterative Nature of the Process', [60, 80, 200])
    bullets(paper.iterative_nature)
  }

  // ============================================================
  // CHALLENGES
  // ============================================================
  if ((paper.challenges || []).length) {
    h2('Challenges in Educational Policymaking', [180, 80, 30])
    bullets(paper.challenges)
  }

  // ============================================================
  // STRATEGIES
  // ============================================================
  if ((paper.strategies || []).length) {
    h2('Strategies for Effective Implementation', [40, 140, 90])
    bullets(paper.strategies)
  }

  // ============================================================
  // TAKEAWAY
  // ============================================================
  if (paper.takeaway && paper.takeaway.trim()) {
    h2('Key Takeaway', [40, 80, 160])
    paragraph(paper.takeaway)
  }

  // ============================================================
  // APPENDIX
  // ============================================================
  ensureSpace(80)
  h2('Appendix · Source Deliberation Snapshot', [80, 90, 110])
  pdf.setFont('times', 'italic')
  pdf.setFontSize(9.5)
  pdf.setTextColor(80, 90, 110)
  pdf.text(
    'The brief above synthesises the swarm-of-swarms deliberation summarised below.',
    M, y,
  )
  y += 14
  drawDeliberationAppendix()

  drawFooter()

  const ts = new Date(report.timestamp).toISOString().replace(/[:.]/g, '-').slice(0, 19)
  pdf.save(`vishwamitra-policy-brief-${ts}.pdf`)


  // ----------------------------------------------------------------
  // Component renderers
  // ----------------------------------------------------------------
  function diagnosticTable(items) {
    const colW = [W * 0.26, W * 0.16, W * 0.58]
    const headH = 22
    const cellPad = 8
    pdf.setFontSize(10)

    // Pre-measure rows
    const rowHeights = items.map((it) => {
      pdf.setFont('times', 'bold')
      const a = pdf.splitTextToSize(it.metric || '', colW[0] - 2 * cellPad)
      pdf.setFont('times', 'bold')
      const b = pdf.splitTextToSize(it.value || '', colW[1] - 2 * cellPad)
      pdf.setFont('times', 'normal')
      const c = pdf.splitTextToSize(it.interpretation || '', colW[2] - 2 * cellPad)
      return Math.max(a.length, b.length, c.length) * 13 + 2 * cellPad
    })
    const total = headH + rowHeights.reduce((s, h) => s + h, 0)
    ensureSpace(total + 6)

    // Header
    pdf.setFillColor(248, 250, 254)
    pdf.rect(M, y, W, headH, 'F')
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(160)
    pdf.rect(M, y, W, headH, 'S')
    pdf.line(M + colW[0], y, M + colW[0], y + headH)
    pdf.line(M + colW[0] + colW[1], y, M + colW[0] + colW[1], y + headH)
    pdf.setFont('times', 'bold')
    pdf.setFontSize(10)
    pdf.setTextColor(30, 40, 60)
    pdf.text('Metric', M + cellPad, y + 14)
    pdf.text('Value', M + colW[0] + cellPad, y + 14)
    pdf.text('Interpretation', M + colW[0] + colW[1] + cellPad, y + 14)
    let yy = y + headH

    items.forEach((it, i) => {
      const h = rowHeights[i]
      if (i % 2 === 1) {
        pdf.setFillColor(252, 252, 254)
        pdf.rect(M, yy, W, h, 'F')
      }
      pdf.setLineWidth(0.4)
      pdf.setDrawColor(170)
      pdf.rect(M, yy, W, h, 'S')
      pdf.line(M + colW[0], yy, M + colW[0], yy + h)
      pdf.line(M + colW[0] + colW[1], yy, M + colW[0] + colW[1], yy + h)

      pdf.setFont('times', 'bold')
      pdf.setFontSize(9.5)
      pdf.setTextColor(20, 30, 50)
      ;(pdf.splitTextToSize(it.metric || '', colW[0] - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + cellPad, yy + cellPad + 10 + k * 13))

      pdf.setFont('times', 'bold')
      pdf.setFontSize(10)
      pdf.setTextColor(40, 80, 160)
      ;(pdf.splitTextToSize(it.value || '', colW[1] - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + colW[0] + cellPad, yy + cellPad + 10 + k * 13))

      pdf.setFont('times', 'normal')
      pdf.setFontSize(10)
      pdf.setTextColor(45, 50, 70)
      ;(pdf.splitTextToSize(it.interpretation || '', colW[2] - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + colW[0] + colW[1] + cellPad, yy + cellPad + 10 + k * 13))

      yy += h
    })
    y = yy + 10
  }

  function drawRoadmapPhase(phase) {
    const padding = 12
    const lh = 13
    const innerW = W - 2 * padding

    // Pre-compute heights
    const subSections = [
      ['ACTIONS',      phase.actions || []],
      ['OWNERS',       phase.owners || []],
      ['MILESTONES',   phase.milestones || []],
      ['DEPENDENCIES', phase.dependencies || []],
    ].filter(([, arr]) => arr.length > 0)

    let h = padding + 22  // header
    if (phase.objective) {
      const oLines = pdf.splitTextToSize(phase.objective, innerW)
      h += oLines.length * lh + 6
    }
    for (const [, items] of subSections) {
      h += 14  // subhead
      for (const it of items) {
        const lines = pdf.splitTextToSize(String(it), innerW - 14)
        h += lines.length * lh + 2
      }
      h += 4
    }
    h += padding

    ensureSpace(h + 12)

    // Outer box
    pdf.setFillColor(244, 250, 246)
    pdf.setDrawColor(40, 140, 90)
    pdf.setLineWidth(0.6)
    pdf.rect(M, y, W, h, 'FD')
    pdf.setFillColor(40, 140, 90)
    pdf.rect(M, y, 4, h, 'F')

    let yy = y + padding

    // Header
    pdf.setFont('times', 'bold')
    pdf.setFontSize(12)
    pdf.setTextColor(20, 90, 50)
    pdf.text(phase.phase_name || '', M + padding, yy + 4)
    pdf.setFont('times', 'italic')
    pdf.setFontSize(10)
    pdf.setTextColor(80, 100, 90)
    pdf.text(phase.window || '', PW - M - padding, yy + 4, { align: 'right' })
    yy += 18

    // Objective
    if (phase.objective) {
      pdf.setFont('times', 'italic')
      pdf.setFontSize(10)
      pdf.setTextColor(40, 50, 60)
      const oLines = pdf.splitTextToSize(phase.objective, innerW)
      oLines.forEach((ln) => {
        pdf.text(ln, M + padding, yy + 2)
        yy += lh
      })
      yy += 4
    }

    // Sub-sections
    for (const [label, items] of subSections) {
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(8)
      pdf.setTextColor(40, 100, 70)
      pdf.text(label, M + padding, yy + 2)
      yy += 10
      pdf.setFont('times', 'normal')
      pdf.setFontSize(9.5)
      pdf.setTextColor(40, 50, 60)
      for (const it of items) {
        const lines = pdf.splitTextToSize(String(it), innerW - 14)
        pdf.text('•', M + padding, yy + 2)
        lines.forEach((ln, k) => {
          pdf.text(ln, M + padding + 12, yy + 2 + k * lh)
        })
        yy += lines.length * lh + 2
      }
      yy += 4
    }

    y += h + 10
  }

  function drawRiskTable(items) {
    const colW = [W * 0.32, W * 0.12, W * 0.12, W * 0.44]
    const headH = 22
    const cellPad = 8
    pdf.setFontSize(10)

    const rowHeights = items.map((it) => {
      pdf.setFont('times', 'bold')
      const a = pdf.splitTextToSize(it.risk || '', colW[0] - 2 * cellPad)
      pdf.setFont('times', 'normal')
      const d = pdf.splitTextToSize(it.mitigation || '', colW[3] - 2 * cellPad)
      return Math.max(a.length, d.length, 1) * 13 + 2 * cellPad
    })
    const total = headH + rowHeights.reduce((s, h) => s + h, 0)
    ensureSpace(total + 6)

    pdf.setFillColor(252, 244, 244)
    pdf.rect(M, y, W, headH, 'F')
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(180, 100, 100)
    pdf.rect(M, y, W, headH, 'S')
    let cx = M
    for (let i = 0; i < colW.length - 1; i++) {
      cx += colW[i]
      pdf.line(cx, y, cx, y + headH)
    }
    pdf.setFont('times', 'bold')
    pdf.setFontSize(10)
    pdf.setTextColor(120, 30, 30)
    pdf.text('Risk', M + cellPad, y + 14)
    pdf.text('Likelihood', M + colW[0] + cellPad, y + 14)
    pdf.text('Impact', M + colW[0] + colW[1] + cellPad, y + 14)
    pdf.text('Mitigation', M + colW[0] + colW[1] + colW[2] + cellPad, y + 14)
    let yy = y + headH

    items.forEach((it, i) => {
      const h = rowHeights[i]
      if (i % 2 === 1) {
        pdf.setFillColor(254, 252, 252)
        pdf.rect(M, yy, W, h, 'F')
      }
      pdf.setLineWidth(0.4)
      pdf.setDrawColor(180, 130, 130)
      pdf.rect(M, yy, W, h, 'S')
      let cx2 = M
      for (let i2 = 0; i2 < colW.length - 1; i2++) {
        cx2 += colW[i2]
        pdf.line(cx2, yy, cx2, yy + h)
      }

      // Risk text
      pdf.setFont('times', 'bold')
      pdf.setFontSize(9.5)
      pdf.setTextColor(20, 30, 50)
      ;(pdf.splitTextToSize(it.risk || '', colW[0] - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + cellPad, yy + cellPad + 10 + k * 13))

      // Likelihood pill
      drawTierPill(M + colW[0] + cellPad, yy + cellPad + 4, it.likelihood)
      // Impact pill
      drawTierPill(M + colW[0] + colW[1] + cellPad, yy + cellPad + 4, it.impact)

      // Mitigation
      pdf.setFont('times', 'normal')
      pdf.setFontSize(10)
      pdf.setTextColor(45, 50, 70)
      ;(pdf.splitTextToSize(it.mitigation || '', colW[3] - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + colW[0] + colW[1] + colW[2] + cellPad, yy + cellPad + 10 + k * 13))

      yy += h
    })
    y = yy + 10
  }

  function drawTierPill(x, yPos, level) {
    const tier = (level || 'medium').toLowerCase()
    const c = TIER_COLORS[tier] || TIER_COLORS.medium
    const w = 56, h = 14
    pdf.setFillColor(...c)
    pdf.rect(x, yPos, w, h, 'F')
    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(8)
    pdf.setTextColor(255, 255, 255)
    pdf.text(tier.toUpperCase(), x + w / 2, yPos + 10, { align: 'center' })
  }

  function drawFeedbackCard(fb) {
    const padding = 11
    const lh = 13
    const innerW = W - 2 * padding
    const roleColor = ({
      student:     [56, 130, 200],
      teacher:     [200, 140, 30],
      admin:       [140, 90, 200],
      policymaker: [40, 140, 90],
    }[(fb.role || '').toLowerCase()] || [80, 90, 110])

    let h = padding + 18  // name + role
    if (fb.key_concern) {
      h += pdf.splitTextToSize(fb.key_concern || '', innerW).length * lh + 4
    }
    if (fb.direct_quote) {
      h += pdf.splitTextToSize('“' + fb.direct_quote + '”', innerW - 16).length * lh + 4
    }
    if (fb.actionable_request) {
      h += pdf.splitTextToSize(fb.actionable_request || '', innerW - 14).length * lh + 4
    }
    h += padding

    ensureSpace(h + 8)

    pdf.setFillColor(252, 252, 254)
    pdf.setDrawColor(220)
    pdf.setLineWidth(0.4)
    pdf.rect(M, y, W, h, 'FD')
    pdf.setFillColor(...roleColor)
    pdf.rect(M, y, 3, h, 'F')

    let yy = y + padding

    pdf.setFont('times', 'bold')
    pdf.setFontSize(11)
    pdf.setTextColor(20, 30, 50)
    pdf.text(fb.persona_name || '(unnamed)', M + padding, yy + 2)
    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(8)
    pdf.setTextColor(...roleColor)
    pdf.text((fb.role || '').toUpperCase(), PW - M - padding, yy + 2, { align: 'right' })
    yy += 16

    if (fb.key_concern) {
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(7.5)
      pdf.setTextColor(120, 130, 150)
      pdf.text('PRIMARY CONCERN', M + padding, yy)
      yy += 9
      pdf.setFont('times', 'normal')
      pdf.setFontSize(10)
      pdf.setTextColor(45, 50, 70)
      ;(pdf.splitTextToSize(fb.key_concern, innerW))
        .forEach((ln) => { pdf.text(ln, M + padding, yy + 2); yy += lh })
      yy += 4
    }

    if (fb.direct_quote) {
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(7.5)
      pdf.setTextColor(120, 130, 150)
      pdf.text('IN THEIR OWN WORDS', M + padding, yy)
      yy += 9
      // Indented italic quote with vertical bar
      pdf.setDrawColor(...roleColor)
      pdf.setLineWidth(1.2)
      const quoteLines = pdf.splitTextToSize('“' + fb.direct_quote + '”', innerW - 16)
      const qStart = yy
      pdf.setFont('times', 'italic')
      pdf.setFontSize(10)
      pdf.setTextColor(50, 60, 80)
      quoteLines.forEach((ln) => { pdf.text(ln, M + padding + 12, yy + 2); yy += lh })
      pdf.line(M + padding + 4, qStart - 6, M + padding + 4, yy - lh + 4)
      yy += 4
    }

    if (fb.actionable_request) {
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(7.5)
      pdf.setTextColor(120, 130, 150)
      pdf.text('ACTIONABLE REQUEST', M + padding, yy)
      yy += 9
      pdf.setFont('times', 'normal')
      pdf.setFontSize(10)
      pdf.setTextColor(...roleColor)
      ;(pdf.splitTextToSize('→ ' + fb.actionable_request, innerW - 14))
        .forEach((ln) => { pdf.text(ln, M + padding, yy + 2); yy += lh })
    }

    y += h + 8
  }

  function table(headers, rows) {
    if (!rows || !rows.length) return
    const cellPad = 8
    const col1W = Math.max(140, W * 0.32)
    const col2W = W - col1W
    pdf.setFontSize(10)
    const rowHeights = rows.map(([a, b]) => {
      pdf.setFont('times', 'bold')
      const aLines = pdf.splitTextToSize(a || '', col1W - 2 * cellPad)
      pdf.setFont('times', 'normal')
      const bLines = pdf.splitTextToSize(b || '', col2W - 2 * cellPad)
      return Math.max(aLines.length, bLines.length) * 13 + 2 * cellPad
    })
    const headerH = 22
    const total = headerH + rowHeights.reduce((a, b) => a + b, 0)
    ensureSpace(total + 6)

    pdf.setFillColor(245, 245, 248)
    pdf.rect(M, y, W, headerH, 'F')
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(160)
    pdf.rect(M, y, W, headerH, 'S')
    pdf.line(M + col1W, y, M + col1W, y + headerH)
    pdf.setFont('times', 'bold')
    pdf.setFontSize(10)
    pdf.setTextColor(15, 23, 42)
    pdf.text(headers[0], M + cellPad, y + 14)
    pdf.text(headers[1], M + col1W + cellPad, y + 14)
    let yy = y + headerH

    rows.forEach(([a, b], i) => {
      const h = rowHeights[i]
      if (i % 2 === 1) {
        pdf.setFillColor(252, 252, 254)
        pdf.rect(M, yy, W, h, 'F')
      }
      pdf.setLineWidth(0.4)
      pdf.setDrawColor(160)
      pdf.rect(M, yy, W, h, 'S')
      pdf.line(M + col1W, yy, M + col1W, yy + h)

      pdf.setFont('times', 'bold')
      pdf.setFontSize(10)
      pdf.setTextColor(20, 30, 50)
      ;(pdf.splitTextToSize(a || '', col1W - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + cellPad, yy + cellPad + 10 + k * 13))

      pdf.setFont('times', 'normal')
      pdf.setTextColor(40, 45, 60)
      ;(pdf.splitTextToSize(b || '', col2W - 2 * cellPad))
        .forEach((ln, k) => pdf.text(ln, M + col1W + cellPad, yy + cellPad + 10 + k * 13))
      yy += h
    })
    y = yy + 6
  }

  function drawDeliberationAppendix() {
    const final = report.final_action || []
    const reson = report.resonance_per_intervention || []
    const flagSet = new Set(report.dissonance_flags || [])
    const rowH = 16
    const headH = 20
    const colW = [W * 0.30, W * 0.35, W * 0.35]
    ensureSpace(headH + ACTION_NAMES.length * rowH + 8)
    pdf.setFillColor(245, 245, 248)
    pdf.rect(M, y, W, headH, 'F')
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(160)
    pdf.rect(M, y, W, headH, 'S')
    pdf.setFont('times', 'bold')
    pdf.setFontSize(9.5)
    pdf.setTextColor(15, 23, 42)
    pdf.text('Intervention', M + 8, y + 14)
    pdf.text('Recommended intensity', M + colW[0] + 8, y + 14)
    pdf.text('Cross-swarm resonance', M + colW[0] + colW[1] + 8, y + 14)
    let yy = y + headH

    ACTION_NAMES.forEach((n, i) => {
      pdf.setLineWidth(0.3)
      pdf.setDrawColor(200)
      pdf.line(M, yy, M + W, yy)
      pdf.line(M + colW[0], yy, M + colW[0], yy + rowH)
      pdf.line(M + colW[0] + colW[1], yy, M + colW[0] + colW[1], yy + rowH)

      const isFlag = flagSet.has(n)
      pdf.setFont('times', isFlag ? 'bold' : 'normal')
      pdf.setFontSize(9.5)
      pdf.setTextColor(isFlag ? 180 : 40, isFlag ? 30 : 45, isFlag ? 30 : 60)
      pdf.text(n, M + 8, yy + 12)

      const f = Math.max(0, Math.min(1, final[i] ?? 0))
      pdf.setFillColor(228, 232, 240)
      pdf.rect(M + colW[0] + 8, yy + 4, colW[1] - 60, 7, 'F')
      pdf.setFillColor(40, 80, 160)
      pdf.rect(M + colW[0] + 8, yy + 4, (colW[1] - 60) * f, 7, 'F')
      pdf.setFont('times', 'normal')
      pdf.setTextColor(40)
      pdf.text(f.toFixed(2), M + colW[0] + colW[1] - 6, yy + 12, { align: 'right' })

      const r = Math.max(0, Math.min(1, reson[i] ?? 0))
      pdf.setFillColor(228, 232, 240)
      pdf.rect(M + colW[0] + colW[1] + 8, yy + 4, colW[2] - 60, 7, 'F')
      if (r > 0.75)      pdf.setFillColor(40, 140, 70)
      else if (r > 0.55) pdf.setFillColor(200, 140, 30)
      else               pdf.setFillColor(190, 50, 50)
      pdf.rect(M + colW[0] + colW[1] + 8, yy + 4, (colW[2] - 60) * r, 7, 'F')
      pdf.setTextColor(40)
      pdf.text(r.toFixed(2), M + W - 6, yy + 12, { align: 'right' })

      yy += rowH
    })
    pdf.line(M, yy, M + W, yy)
    pdf.line(M, y, M, yy)
    pdf.line(M + W, y, M + W, yy)
    y = yy + 12

    if (flagSet.size > 0) {
      pdf.setFont('times', 'italic')
      pdf.setFontSize(9.5)
      pdf.setTextColor(180, 30, 30)
      pdf.text(`Dissonance flags: ${[...flagSet].join(', ')}`, M, y)
      y += 14
    }
  }
}
