import React, { useState } from 'react';
import axios from 'axios';
import { Zap, AlertTriangle, CheckCircle, XCircle, Send, TrendingUp, ArrowRight, HelpCircle, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react';
import { FadeIn } from '../components/Animations';

const FEATURE_RANGES = {
  f0: [-3, 3], f1: [-3, 3], f2: [-3, 3], f3: [-5, 5], f4: [0, 25],
  f5: [-2, 2], f6: [-2, 2], f7: [-2, 2], f8: [-2, 2], f9: [-2, 2],
  f10: [-2, 2], f11: [-2, 2],
};

// Feature descriptions for the help panel
const FEATURE_DOCS = [
  { key: 'f0', label: 'Browsing Recency',     range: '-3 to 3',  hint: 'Higher = more recently active. Positive values indicate recent site visits.' },
  { key: 'f1', label: 'Engagement Depth',     range: '-3 to 3',  hint: 'Higher = deeper page engagement. Correlates with interest signals.' },
  { key: 'f2', label: 'Return Frequency',     range: '-3 to 3',  hint: 'Negative values = infrequent visitor. Positive = habitual returnee.' },
  { key: 'f3', label: 'Purchase Intent',      range: '-5 to 5',  hint: 'Key feature. Negative = high purchase intent. Positive = browsing only.' },
  { key: 'f4', label: 'Historical Activity',  range: '0 to 25',  hint: 'Key feature. Higher = more active historical record. Persuadables cluster above 10.' },
  { key: 'f5', label: 'Category Affinity A',  range: '-2 to 2',  hint: 'Product category signal. Positive = strong affinity in this segment.' },
  { key: 'f6', label: 'Category Affinity B',  range: '-2 to 2',  hint: 'Secondary category signal. Higher = broader category engagement.' },
  { key: 'f7', label: 'Session Intensity',    range: '-2 to 2',  hint: 'Click depth per session. Higher = more deliberate browsing behavior.' },
  { key: 'f8', label: 'Cross-Device Signal',  range: '-2 to 2',  hint: 'Multi-device usage pattern. Positive = cross-device user.' },
  { key: 'f9', label: 'Recency Decay',        range: '-2 to 2',  hint: 'Recency-weighted engagement score. Higher = recently re-engaged.' },
  { key: 'f10', label: 'Cart Abandon Rate',   range: '-2 to 2',  hint: 'Cart abandonment signal. Negative = tendency to abandon without converting.' },
  { key: 'f11', label: 'Loyalty Signal',      range: '-2 to 2',  hint: 'Repeat purchase / loyalty indicator. Positive = returning buyer.' },
];

const PRESETS = {
  'Persuadable (High Uplift)': { f0: 0.8, f1: 1.2, f2: -0.3, f3: -2.1, f4: 13.5, f5: 0.4, f6: 0.9, f7: 0.1, f8: 0.2, f9: 0.5, f10: -0.2, f11: 0.1 },
  'Lost Cause (Low Uplift)':   { f0: 0.1, f1: -0.2, f2: 0.1, f3: 1.2, f4: 2.0, f5: -0.1, f6: 0.0, f7: -0.1, f8: 0.0, f9: 0.1, f10: 0.0, f11: -0.1 },
  'Sure Thing':                { f0: 0.5, f1: 0.8, f2: 0.4, f3: 0.2, f4: 8.0, f5: 0.3, f6: 0.5, f7: 0.2, f8: 0.1, f9: 0.3, f10: 0.1, f11: 0.0 },
};

const PIPELINE_STEPS = [
  { box: 'amber',   icon: '🎯', name: 'X-Learner',    sub: '5 LightGBM' },
  { arrow: true },
  { box: 'violet',  icon: '🧠', name: 'Distillation', sub: 'Soft labels' },
  { arrow: true },
  { box: 'emerald', icon: '⚡', name: 'Student Tree', sub: 'depth=5, <1ms' },
  { arrow: true },
  { box: 'cyan',    icon: '💰', name: 'Bid Policy',   sub: 'LTV x CATE > Cost' },
];

const initialFeatures = () => Object.fromEntries(Object.keys(FEATURE_RANGES).map(k => [k, 0.1]));

const LTV = 10.0;
const COST = 0.10;
const IMPRESSIONS_AT_SCALE = 10_000_000;

export default function InferenceEngine() {
  const [features, setFeatures] = useState(initialFeatures());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [activePreset, setActivePreset] = useState(null);
  const [showHelp, setShowHelp] = useState(false);
  // dirty = features changed since last inference run
  const [dirty, setDirty] = useState(false);

  const handleChange = (key, val) => {
    setFeatures(prev => ({ ...prev, [key]: Number(val) }));
    setActivePreset(null);
    setDirty(true);
  };

  const loadPreset = (name) => {
    setFeatures({ ...PRESETS[name] });
    setActivePreset(name);
    setError(null);
    setDirty(true); // mark dirty so user knows to re-run
  };

  const handleReset = () => {
    setFeatures(initialFeatures());
    setActivePreset(null);
    setResult(null);
    setError(null);
    setDirty(false);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null); // clear previous result so UI shows loading state clearly
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://a-b-testing-569o.onrender.com';
      const res = await axios.post(`${apiUrl}/predict`, features, { timeout: 15000 });
      const entry = {
        ...res.data,
        timestamp: new Date().toLocaleTimeString(),
        features: { ...features },
      };
      setResult(entry);
      setHistory(h => [entry, ...h].slice(0, 6));
      setDirty(false);
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Could not connect to the inference API. Please ensure the backend service is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <FadeIn>
        <div className="page-header">
          <div className="page-eyebrow"><span>▸</span> Phase 4 of 4 · Inference Engine</div>
          <h1>Real-Time <span className="gradient-text">Bid</span> Decision</h1>
          <p>
            The Distilled Student Decision Tree runs in{' '}
            <strong style={{ color: 'var(--accent-emerald)' }}>&lt;1ms</strong>,
            production-ready for Real-Time Bidding environments. Enter user context to
            get the causal uplift prediction and bidding decision instantly.
          </p>
        </div>
      </FadeIn>

      {/* Narrative Banner */}
      <FadeIn delay={60}>
        <div className="narrative-banner">
          <div className="narrative-banner-label">🚀 The Payoff</div>
          <div className="narrative-banner-text">
            The complex X-Learner ensemble compressed to a depth-5 Decision Tree with{' '}
            <strong style={{ color: 'var(--accent-amber-light)' }}>65.5% R2 fidelity</strong>.
            At 10M impressions, this policy generates an estimated{' '}
            <strong style={{ color: 'var(--accent-emerald)' }}>
              ${((0.083) * IMPRESSIONS_AT_SCALE / 1_000_000).toFixed(1)}M
            </strong>{' '}
            incremental profit. Deploy today.
          </div>
        </div>
      </FadeIn>

      {/* Pipeline Architecture */}
      <FadeIn delay={80}>
        <div className="glass-card card-pad" style={{ marginBottom: 24 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Model Pipeline Architecture</div>
              <div className="card-subtitle">From teacher ensemble to production decision in 4 stages</div>
            </div>
            <span className="card-badge amber">Knowledge Distillation</span>
          </div>
          <div className="pipeline-flow">
            {PIPELINE_STEPS.map((step, i) =>
              step.arrow ? (
                <div key={i} className="pipeline-arrow">
                  <ArrowRight size={16} />
                </div>
              ) : (
                <div key={i} className="pipeline-step">
                  <div className={`pipeline-step-box ${step.box}`}>
                    <div className="pipeline-step-icon">{step.icon}</div>
                    <div className="pipeline-step-name">{step.name}</div>
                    <div className="pipeline-step-sub">{step.sub}</div>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      </FadeIn>

      {/* Main Interaction Grid - NOT wrapped in FadeIn to avoid re-render opacity bug */}
      <div className="section-grid-2-1">

        {/* Input Panel */}
        <div className="glass-card card-pad">
          {/* Card Header */}
          <div className="card-header" style={{ marginBottom: 16 }}>
            <div>
              <div className="card-title">User Context Vector</div>
              <div className="card-subtitle">12-dimensional feature space (Criteo anonymized covariates)</div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => setShowHelp(v => !v)}
                title="How to use"
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '4px 10px', borderRadius: 20,
                  border: `1px solid ${showHelp ? 'var(--accent-amber)' : 'rgba(255,255,255,0.1)'}`,
                  background: showHelp ? 'var(--accent-amber-dim)' : 'rgba(255,255,255,0.04)',
                  color: showHelp ? 'var(--accent-amber-light)' : 'var(--text-muted)',
                  fontSize: '0.72rem', fontWeight: 600, cursor: 'pointer',
                  transition: 'all 0.15s', fontFamily: 'var(--font-sans)',
                }}
              >
                <HelpCircle size={13} />
                How to use
                {showHelp ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
              <button
                onClick={handleReset}
                title="Reset all features"
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '4px 10px', borderRadius: 20,
                  border: '1px solid rgba(255,255,255,0.08)',
                  background: 'rgba(255,255,255,0.04)',
                  color: 'var(--text-muted)',
                  fontSize: '0.72rem', fontWeight: 600, cursor: 'pointer',
                  transition: 'all 0.15s', fontFamily: 'var(--font-sans)',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(244,63,94,0.3)'; e.currentTarget.style.color = 'var(--accent-rose)'; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
              >
                <RotateCcw size={12} /> Reset
              </button>
            </div>
          </div>

          {/* Help Panel */}
          {showHelp && (
            <div style={{
              marginBottom: 20, padding: '16px 18px', borderRadius: 10,
              background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.18)',
            }}>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--accent-amber)', marginBottom: 10 }}>
                How to use this tool
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
                {[
                  { step: '1', text: 'Choose a preset below to load a known user archetype (Persuadable, Lost Cause, or Sure Thing), or drag sliders to define a custom user.' },
                  { step: '2', text: 'Click "Run Real-Time Inference" to send the feature vector to the distilled Student Tree model (<1ms latency).' },
                  { step: '3', text: 'The RTB Decision Engine shows the predicted CATE, the economic calculation, and the final BID or DO NOT BID verdict.' },
                  { step: '4', text: 'Change any slider and click the button again to run a new prediction. Each click is an independent inference call.' },
                ].map(({ step, text }) => (
                  <div key={step} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                    <span style={{
                      flexShrink: 0, width: 20, height: 20, borderRadius: '50%',
                      background: 'var(--accent-amber-dim)', border: '1px solid var(--border-amber)',
                      color: 'var(--accent-amber)', fontSize: '0.65rem', fontWeight: 700,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>{step}</span>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>{text}</span>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--accent-amber)', marginBottom: 8 }}>
                Feature reference
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 12px' }}>
                {FEATURE_DOCS.map(({ key, label, range, hint }) => (
                  <div key={key} style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-violet)', fontWeight: 600 }}>{key}</span>
                    {' '}<span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</span>
                    {' '}<span style={{ color: 'var(--text-muted)', fontSize: '0.67rem' }}>({range})</span>
                    <br />
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem', paddingLeft: 4 }}>{hint}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Presets */}
          <div style={{ marginBottom: 6 }}>
            <div style={{ fontSize: '0.66rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>
              Quick Presets
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {Object.keys(PRESETS).map(name => (
                <button
                  key={name}
                  onClick={() => loadPreset(name)}
                  style={{
                    padding: '6px 14px', borderRadius: 20,
                    border: `1px solid ${activePreset === name ? 'var(--accent-amber)' : 'rgba(255,255,255,0.08)'}`,
                    background: activePreset === name ? 'var(--accent-amber-dim)' : 'rgba(255,255,255,0.04)',
                    color: activePreset === name ? 'var(--accent-amber-light)' : 'var(--text-secondary)',
                    fontSize: '0.75rem', fontWeight: 600, cursor: 'pointer',
                    transition: 'all 0.15s', fontFamily: 'var(--font-sans)',
                  }}
                  onMouseEnter={e => { if (activePreset !== name) { e.currentTarget.style.borderColor = 'rgba(245,158,11,0.3)'; e.currentTarget.style.color = 'var(--accent-amber-light)'; } }}
                  onMouseLeave={e => { if (activePreset !== name) { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = 'var(--text-secondary)'; } }}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>

          <hr style={{ border: 'none', borderTop: '1px solid var(--border-subtle)', margin: '16px 0' }} />

          {/* Feature Sliders */}
          <div style={{ fontSize: '0.66rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12 }}>
            Feature Vector
          </div>
          <div className="form-grid">
            {FEATURE_DOCS.map(({ key, label }) => {
              const val = features[key];
              const [min, max] = FEATURE_RANGES[key];
              return (
                <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <label className="form-label" title={label}>
                    {key.toUpperCase()}
                    <span style={{ fontFamily: 'var(--font-sans)', letterSpacing: 0, textTransform: 'none', fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.6rem', marginLeft: 4 }}>
                      {label}
                    </span>
                  </label>
                  <input
                    type="range"
                    min={min} max={max} step={0.05}
                    value={val}
                    onChange={e => handleChange(key, e.target.value)}
                    style={{ width: '100%', marginBottom: 2, accentColor: 'var(--accent-amber)' }}
                  />
                  <input
                    type="number"
                    step="0.05"
                    value={val}
                    onChange={e => handleChange(key, e.target.value)}
                    className="form-input"
                    style={{ padding: '5px 8px', fontSize: '0.78rem' }}
                  />
                </div>
              );
            })}
          </div>

          {/* Run Button */}
          <button
            onClick={handlePredict}
            disabled={loading}
            className="btn-primary"
            style={{ width: '100%', marginTop: 16, position: 'relative' }}
          >
            {loading ? (
              <>
                <div style={{ width: 16, height: 16, border: '2px solid rgba(10,13,20,0.3)', borderTopColor: '#0A0D14', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
                Running Inference...
              </>
            ) : dirty && result ? (
              <><RotateCcw size={15} /> Re-run Inference</>
            ) : (
              <><Send size={15} /> Run Real-Time Inference</>
            )}
          </button>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

          {/* Dirty state notice */}
          {dirty && result && !loading && (
            <div style={{ marginTop: 8, textAlign: 'center', fontSize: '0.7rem', color: 'var(--accent-amber)', opacity: 0.8 }}>
              Features changed since last run. Click above to update the prediction.
            </div>
          )}

          {error && (
            <div className="alert-error" style={{ marginTop: 12 }}>
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>
                {error}
                <br />
                <span style={{ color: 'var(--text-muted)', fontSize: '0.73rem' }}>
                  Start the API:{' '}
                  <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)' }}>
                    uvicorn api.main:app --port 8080
                  </code>
                </span>
              </span>
            </div>
          )}
        </div>

        {/* Decision + Forecast Panel - also NOT in FadeIn to avoid re-render issue */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* RTB Decision */}
          <div className="glass-card card-pad">
            <div className="card-title" style={{ marginBottom: 20 }}>RTB Decision Engine</div>
            <div className="decision-panel">
              {loading ? (
                <div className="decision-idle">
                  <div style={{ width: 48, height: 48, borderRadius: '50%', border: '3px solid rgba(245,158,11,0.2)', borderTopColor: 'var(--accent-amber)', animation: 'spin 0.9s linear infinite' }} />
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Computing uplift...</p>
                </div>
              ) : !result ? (
                <div className="decision-idle">
                  <div className="decision-idle-icon">
                    <Zap size={28} color="var(--text-muted)" />
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', lineHeight: 1.6 }}>
                    Select a preset or adjust sliders,<br />then run inference to see the decision.
                  </p>
                </div>
              ) : (
                <>
                  <div style={{ textAlign: 'center', width: '100%' }}>
                    <div className="score-label">Predicted CATE (Uplift)</div>
                    <div
                      className="score-number"
                      style={{ color: result.uplift_score > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}
                    >
                      {result.uplift_score > 0 ? '+' : ''}{(result.uplift_score * 100).toFixed(5)}%
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: 4 }}>
                      Expected incremental conversion rate
                    </div>
                  </div>

                  {/* Economic Breakdown */}
                  <div style={{ width: '100%', padding: '14px 16px', borderRadius: 10, background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                      Economic Calculation
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      {[
                        { label: 'CATE x LTV', val: `$${(result.uplift_score * LTV).toFixed(5)}` },
                        { label: 'Ad Cost',    val: `$${COST.toFixed(2)}` },
                        { label: 'Net Value',  val: `$${((result.uplift_score * LTV) - COST).toFixed(5)}`, highlight: true },
                        { label: 'Decision',   val: result.bid_eligible ? 'BID ✓' : 'SKIP ✗', highlight: true },
                      ].map(({ label, val, highlight }) => (
                        <div key={label} style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(255,255,255,0.05)' }}>
                          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: 2, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{label}</div>
                          <div style={{ fontSize: '0.875rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: highlight ? (result.bid_eligible ? 'var(--accent-emerald)' : 'var(--accent-rose)') : 'var(--text-primary)' }}>
                            {val}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className={`decision-verdict ${result.bid_eligible ? 'bid' : 'no-bid'}`}>
                    <div className="decision-verdict-label">Bandit Policy Action</div>
                    <div className="decision-verdict-text">
                      {result.bid_eligible
                        ? <><CheckCircle size={20} style={{ display: 'inline', marginRight: 8, verticalAlign: 'text-bottom' }} />BID</>
                        : <><XCircle size={20} style={{ display: 'inline', marginRight: 8, verticalAlign: 'text-bottom' }} />DO NOT BID</>
                      }
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Forecast Card */}
          <div className="forecast-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <TrendingUp size={16} color="var(--accent-emerald)" />
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Profit Forecast at Scale
              </span>
            </div>
            {[
              { scale: '1M',   profit: '$' + (0.083 * 1_000_000   / 1_000).toFixed(0)   + 'K' },
              { scale: '10M',  profit: '$' + (0.083 * 10_000_000  / 1_000_000).toFixed(2) + 'M' },
              { scale: '100M', profit: '$' + (0.083 * 100_000_000 / 1_000_000).toFixed(1) + 'M' },
            ].map(row => (
              <div key={row.scale} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{row.scale} impressions</span>
                <span style={{ fontSize: '0.9rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>+{row.profit}</span>
              </div>
            ))}
            <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 10, lineHeight: 1.5 }}>
              Estimated at $0.083/user bandit profit rate (current pipeline performance).
              Assumes stable feature distribution.
            </p>
          </div>

          {/* Recent Predictions History */}
          {history.length > 0 && (
            <div className="glass-card card-pad">
              <div className="card-title" style={{ marginBottom: 14, fontSize: '0.85rem' }}>Recent Predictions</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {history.map((h, i) => (
                  <div key={`${h.timestamp}-${i}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', borderRadius: 8, background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {h.bid_eligible
                        ? <CheckCircle size={14} color="var(--accent-emerald)" />
                        : <XCircle size={14} color="var(--accent-rose)" />
                      }
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                        {h.uplift_score > 0 ? '+' : ''}{(h.uplift_score * 100).toFixed(5)}%
                      </span>
                    </div>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {h.timestamp}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
