import { SLIDER_FIELDS } from '../../lib/swarm-api.js'

export default function StateEditor({ state, scenario, onStateChange, onScenarioChange, onRun, loading, error }) {
  function handleSlider(key, value) {
    onStateChange({ ...state, [key]: Number(value) })
  }

  return (
    <>
      <div className="vm-section-h">State Vector</div>
      <div className="vm-editor">
        {SLIDER_FIELDS.map((f) => {
          const v = state[f.key] ?? 0
          // For "invert" fields (dropout, burnout) high values are bad → color the readout red.
          const isBad = f.invert && v > 0.5
          const pct = ((v - f.min) / (f.max - f.min)) * 100
          return (
            <div key={f.key} className="vm-slider-row">
              <div className="vm-slider-label">
                <span className="name">{f.label}</span>
                <span className={`val ${isBad ? 'bad' : ''}`}>{f.format(v)}</span>
              </div>
              <input
                type="range"
                className="vm-slider"
                min={f.min}
                max={f.max}
                step={f.step}
                value={v}
                onChange={(e) => handleSlider(f.key, e.target.value)}
                style={{
                  background: `linear-gradient(to right,
                    var(--vm-accent) 0%, var(--vm-accent) ${pct}%,
                    var(--vm-bg-elev) ${pct}%, var(--vm-bg-elev) 100%)`,
                }}
              />
            </div>
          )
        })}
        <div className="vm-slider-row">
          <div className="vm-slider-label">
            <span className="name">Scenario brief</span>
          </div>
        </div>
        <div className="vm-scenario-area">
          <textarea
            value={scenario}
            onChange={(e) => onScenarioChange(e.target.value)}
            placeholder="Describe the situation the swarms must deliberate on…"
            spellCheck={false}
          />
        </div>
        {error && <div className="vm-error">{error}</div>}
      </div>
      <div className="vm-run-row">
        <button
          className={`vm-run-btn ${loading ? 'loading' : ''}`}
          onClick={onRun}
          disabled={loading}
        >
          {loading ? '◐ Deliberating…' : '▶ Run Deliberation'}
        </button>
      </div>
    </>
  )
}
