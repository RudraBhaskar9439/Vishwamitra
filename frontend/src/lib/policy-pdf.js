// IEEE-style PDF renderer for Vishwamitra policy reports.
//
// Input:  { paper, report } where `paper` is the LLM-generated structured
//         prose returned by POST /swarms/policy-report, and `report` is the
//         original ResonanceReport JSON from /swarms/deliberate.
// Output: triggers a browser download of vishwamitra-ieee-policy-<ts>.pdf.
//
// Style choices:
//   - Times serif (jsPDF built-in 'times')
//   - Single column at letter-paper width — easier to read than two-column
//     in modern policy PDFs and avoids two-column line-justification pain.
//   - Centered title / authors / affiliation block.
//   - Bold "Abstract—" inline prefix; italic body.
//   - Bold "Index Terms—" inline prefix; italic body.
//   - Roman-numeral sections (I., II., III., ...).
//   - Page footer with running title + page number.
//   - One inline figure (Fig. 1) showing the recommended action vector and
//     per-intervention resonance.

import jsPDF from 'jspdf'

const ACTION_NAMES = [
  'funding_boost','teacher_incentive','student_scholarship','attendance_mandate',
  'resource_realloc','transparency_report','staff_hiring','counseling_programs',
]

export function renderIEEEPolicyPDF({ paper, report }) {
  const pdf = new jsPDF({ unit: 'pt', format: 'letter' })
  const PW = pdf.internal.pageSize.getWidth()
  const PH = pdf.internal.pageSize.getHeight()
  const M  = 64
  const W  = PW - 2 * M

  let y = M
  let pageNum = 1
  const runningTitle = (paper.title || 'Policy Deliberation Report').slice(0, 80)

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

  // ---------------------- TITLE BLOCK ----------------------
  pdf.setFont('times', 'bold')
  pdf.setFontSize(17)
  pdf.setTextColor(15, 23, 42)
  const titleLines = pdf.splitTextToSize(paper.title || 'Untitled Policy Report', W)
  for (const line of titleLines) {
    pdf.text(line, PW / 2, y, { align: 'center' })
    y += 21
  }
  y += 4

  pdf.setFont('times', 'normal')
  pdf.setFontSize(11)
  pdf.setTextColor(40)
  pdf.text(paper.authors || 'Vishwamitra Swarm Deliberation System', PW / 2, y, { align: 'center' })
  y += 14

  pdf.setFont('times', 'italic')
  pdf.setFontSize(10)
  pdf.setTextColor(70)
  const aff = pdf.splitTextToSize(paper.affiliation || '', W)
  for (const line of aff) {
    pdf.text(line, PW / 2, y, { align: 'center' })
    y += 12
  }
  y += 18

  // ---------------------- ABSTRACT ----------------------
  if (paper.abstract && paper.abstract.trim()) {
    ensureSpace(40)
    const ABS_M = M + 20  // slight indent on both sides for abstract
    const ABS_W = PW - 2 * ABS_M
    pdf.setFont('times', 'bold')
    pdf.setFontSize(9.5)
    pdf.setTextColor(15, 23, 42)
    pdf.text('Abstract—', ABS_M, y)
    const prefixWidth = pdf.getTextWidth('Abstract—')
    pdf.setFont('times', 'italic')
    pdf.setFontSize(9.5)
    pdf.setTextColor(35, 40, 55)

    // First wrap with the prefix offset on line 1; subsequent lines flush left.
    const firstLineWidth = ABS_W - prefixWidth - 4
    const allText = paper.abstract.trim()
    // Word-fit greedy split for the first line:
    const words = allText.split(/\s+/)
    let firstLine = ''
    let i = 0
    while (i < words.length) {
      const trial = firstLine ? firstLine + ' ' + words[i] : words[i]
      if (pdf.getTextWidth(trial) > firstLineWidth) break
      firstLine = trial
      i += 1
    }
    pdf.text(firstLine, ABS_M + prefixWidth + 4, y)
    y += 12
    const remaining = words.slice(i).join(' ')
    if (remaining) {
      const rest = pdf.splitTextToSize(remaining, ABS_W)
      ensureSpace(rest.length * 12)
      pdf.text(rest, ABS_M, y)
      y += rest.length * 12
    }
    y += 6
  }

  // ---------------------- KEYWORDS ----------------------
  if (paper.keywords && paper.keywords.trim()) {
    ensureSpace(20)
    const ABS_M = M + 20
    pdf.setFont('times', 'bold')
    pdf.setFontSize(9.5)
    pdf.setTextColor(15, 23, 42)
    pdf.text('Index Terms—', ABS_M, y)
    const offset = pdf.getTextWidth('Index Terms—')
    pdf.setFont('times', 'italic')
    pdf.setTextColor(35, 40, 55)
    const kwWidth = (PW - 2 * ABS_M) - offset - 4
    const kwLines = pdf.splitTextToSize(paper.keywords.trim(), kwWidth)
    pdf.text(kwLines[0] || '', ABS_M + offset + 4, y)
    y += 12
    for (let k = 1; k < kwLines.length; k++) {
      pdf.text(kwLines[k], ABS_M, y)
      y += 12
    }
    y += 14
  }

  // Section divider rule
  pdf.setLineWidth(0.5)
  pdf.setDrawColor(180)
  pdf.line(M, y, PW - M, y)
  y += 14

  // ---------------------- BODY SECTIONS ----------------------
  const sections = [
    ['I',   'INTRODUCTION',          paper.introduction],
    ['II',  'METHODOLOGY',           paper.methodology],
    ['III', 'RESULTS AND ANALYSIS',  paper.results,           { figureAfter: 1 }],
    ['IV',  'FUTURE PROJECTIONS',    paper.future_projections],
    ['V',   'DISCUSSION',            paper.discussion],
    ['VI',  'CONCLUSION',            paper.conclusion],
  ]

  for (const [num, name, content, opts] of sections) {
    if (!content || !content.trim()) continue

    ensureSpace(32)
    pdf.setFont('times', 'bold')
    pdf.setFontSize(11)
    pdf.setTextColor(15, 23, 42)
    pdf.text(`${num}.  ${name}`, M, y)
    y += 16

    pdf.setFont('times', 'normal')
    pdf.setFontSize(10)
    pdf.setTextColor(28, 32, 48)

    // Split into paragraphs on blank-line OR single-line gap.
    const paras = content
      .split(/\n\s*\n+/)
      .map((p) => p.trim().replace(/\s+/g, ' '))
      .filter(Boolean)

    for (const para of paras) {
      const lines = pdf.splitTextToSize(para, W)
      ensureSpace(lines.length * 12.5 + 4)
      // first line: indent 18pt; subsequent flush left
      lines.forEach((line, idx) => {
        pdf.text(line, idx === 0 ? M + 18 : M, y)
        y += 12.5
      })
      y += 4
    }
    y += 4

    // After section III drop in Figure 1.
    if (opts && opts.figureAfter === 1) {
      drawFigure1(pdf, { y, W, M, report, ensureSpace, getY: () => y, setY: (v) => { y = v } })
    }
  }

  // ---------------------- REFERENCES ----------------------
  ensureSpace(40)
  pdf.setFont('times', 'bold')
  pdf.setFontSize(11)
  pdf.setTextColor(15, 23, 42)
  pdf.text('REFERENCES', M, y)
  y += 16

  const refs = [
    `Vishwamitra Swarm Deliberation System, "Source deliberation report," generated ${new Date(report.timestamp).toLocaleString()}.`,
    'Multi-Agent Policy Analytics Lab, "Cross-swarm Resonance metric," internal note: r_i = 1 − σ_norm(swarm aggregated action_i).',
    'Vishwamitra Project, "Persona configuration: 4 role-swarms × 3 heterogeneous personas," roles.yaml v0.1.',
    'IEEE Standard for Reporting on Multi-Agent Policy Deliberation (illustrative; no formal standard cited).',
  ]
  pdf.setFont('times', 'normal')
  pdf.setFontSize(9)
  pdf.setTextColor(40, 45, 60)
  refs.forEach((r, i) => {
    const tag = `[${i + 1}]`
    const tagW = pdf.getTextWidth(tag) + 6
    const lines = pdf.splitTextToSize(r, W - tagW)
    ensureSpace(lines.length * 11 + 4)
    pdf.text(tag, M, y)
    pdf.text(lines, M + tagW, y)
    y += lines.length * 11 + 4
  })

  drawFooter()

  const ts = new Date(report.timestamp).toISOString().replace(/[:.]/g, '-').slice(0, 19)
  pdf.save(`vishwamitra-policy-${ts}.pdf`)
}


// ---- Fig. 1: dual bar chart, recommended action vector + resonance ----
function drawFigure1(pdf, ctx) {
  const { W, M, report, ensureSpace, getY, setY } = ctx
  const final = report.final_action || []
  const reson = report.resonance_per_intervention || []
  const flags = new Set(report.dissonance_flags || [])

  const figW = W
  const ROW_H = 18
  const HEADER_H = 18
  const figH = HEADER_H + ACTION_NAMES.length * ROW_H + 26
  ensureSpace(figH + 30)

  let y = getY()

  // Caption space
  y += 4
  // Inner border
  pdf.setLineWidth(0.4)
  pdf.setDrawColor(140)
  pdf.rect(M, y, figW, figH, 'S')
  pdf.setFont('times', 'bold')
  pdf.setFontSize(9)
  pdf.setTextColor(20)
  pdf.text('intervention', M + 8, y + 12)
  pdf.text('recommended intensity', M + 160, y + 12)
  pdf.text('cross-swarm resonance', M + figW - 168, y + 12)
  pdf.setLineWidth(0.3)
  pdf.line(M, y + HEADER_H, M + figW, y + HEADER_H)

  pdf.setFont('times', 'normal')
  pdf.setFontSize(9)
  ACTION_NAMES.forEach((n, i) => {
    const yy = y + HEADER_H + i * ROW_H + 12
    const isFlag = flags.has(n)
    pdf.setTextColor(isFlag ? 180 : 40, isFlag ? 30 : 45, isFlag ? 30 : 60)
    pdf.text(n, M + 8, yy)

    // Recommended intensity bar
    const f = Math.max(0, Math.min(1, final[i] ?? 0))
    const barX = M + 160
    const barW = 140
    pdf.setFillColor(228, 232, 240)
    pdf.rect(barX, yy - 8, barW, 8, 'F')
    pdf.setFillColor(40, 80, 160)
    pdf.rect(barX, yy - 8, barW * f, 8, 'F')
    pdf.setTextColor(20)
    pdf.text(f.toFixed(2), barX + barW + 6, yy)

    // Resonance bar
    const r = Math.max(0, Math.min(1, reson[i] ?? 0))
    const rX = M + figW - 168
    const rW = 120
    pdf.setFillColor(228, 232, 240)
    pdf.rect(rX, yy - 8, rW, 8, 'F')
    if (r > 0.75)      pdf.setFillColor(40, 140, 70)   // green
    else if (r > 0.55) pdf.setFillColor(200, 140, 30)  // amber
    else               pdf.setFillColor(190, 50, 50)   // red
    pdf.rect(rX, yy - 8, rW * r, 8, 'F')
    pdf.setTextColor(20)
    pdf.text(r.toFixed(2), rX + rW + 6, yy)
  })

  y += figH + 6
  pdf.setFont('times', 'italic')
  pdf.setFontSize(8.5)
  pdf.setTextColor(80)
  const cap = pdf.splitTextToSize(
    'Fig. 1.  Recommended action vector and cross-swarm Resonance per intervention. '
    + 'Bars on the left encode confidence-weighted recommended intensity in [0,1]; '
    + 'bars on the right encode Resonance, defined as 1 − σ_norm of role-swarm '
    + 'aggregated action across the four swarms. Interventions printed in red are '
    + 'flagged as dissonant (Resonance below 0.55).',
    W
  )
  pdf.text(cap, M, y)
  y += cap.length * 11 + 8
  setY(y)
}
