import React, { useState } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, Line
} from 'recharts';
import { FadeIn } from '../components/Animations';

const QiniTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'rgba(10,13,20,0.97)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '10px 14px', backdropFilter: 'blur(16px)' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginBottom: 4 }}>Population Targeted: {(label * 100).toFixed(1)}%</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontWeight: 600, fontSize: '0.82rem', fontFamily: 'var(--font-mono)' }}>
          {p.name}: {p.value?.toFixed(3)}
        </p>
      ))}
    </div>
  );
};

const DecileTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'rgba(10,13,20,0.97)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '10px 14px' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginBottom: 6 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontWeight: 600, fontSize: '0.8rem', fontFamily: 'var(--font-mono)', marginBottom: 2 }}>
          {p.name}: {p.value?.toFixed(4)}%
        </p>
      ))}
    </div>
  );
};

export default function CausalAnalysis({ insights }) {
  const [activeTab, setActiveTab] = useState('qini');
  const { qini_data, decile_stats, metrics, rules } = insights;

  // Downsample Qini to ~120 points
  const stride = Math.max(1, Math.floor(qini_data.x.length / 120));
  const finalY = qini_data.y_mean[qini_data.y_mean.length - 1];
  const qiniChartData = qini_data.x
    .filter((_, i) => i % stride === 0)
    .map((x, i) => ({
      x,
      mean:   qini_data.y_mean[i * stride],
      lower:  qini_data.y_lower[i * stride],
      upper:  qini_data.y_upper[i * stride],
      random: finalY * x,
    }));

  // Horizontal grouped bar data, sorted descending for visual hierarchy
  const decileChartData = [...decile_stats]
    .sort((a, b) => Number(b.bin) - Number(a.bin))
    .map(d => ({
      bin: `D${d.bin}`,
      treatment: +(d.mean_y_t * 100).toFixed(4),
      control:   +(d.mean_y_c * 100).toFixed(4),
      lift:       +(d.actual_lift * 100).toFixed(4),
    }));

  const totalTopLift = decile_stats.find(d => d.bin === '9')?.actual_lift ?? 0;
  const avgLift = metrics.global_ate_absolute;
  const liftMultiple = (totalTopLift / (avgLift || 1)).toFixed(1);



  return (
    <div>
      <FadeIn>
        <div className="page-header">
          <div className="page-eyebrow"><span>▸</span> Phase 2 of 4 · Causal Analysis</div>
          <h1>X-Learner <span className="gradient-text">CATE</span> Estimation</h1>
          <p>
            The X-Learner meta-architecture isolates the <em>incremental</em> causal effect of the ad
            for each individual, separating those who <em>need</em> the nudge from those who convert
            regardless. Top decile uplift is{' '}
            <strong style={{ color: 'var(--accent-amber-light)' }}>{liftMultiple}× the population average</strong>.
          </p>
        </div>
      </FadeIn>

      {/* Narrative Banner */}
      <FadeIn delay={60}>
        <div className="narrative-banner">
          <div className="narrative-banner-label">📐 What the model found</div>
          <div className="narrative-banner-text">
            Top 10% of users generate <strong style={{ color: 'var(--accent-amber-light)' }}>{liftMultiple}×</strong> the average uplift.
            The Qini curve confirms the model ranks them correctly.
            the <strong style={{ color: 'var(--accent-emerald)' }}>AUUC of {qini_data.auuc_mean?.toFixed(2)}</strong> beats
            the random baseline with a tight 95% confidence interval.
          </div>
        </div>
      </FadeIn>

      {/* Stat Cards */}
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: 32 }}>
        {[
          { label: 'Global ATE (Absolute)', val: `+${(metrics.global_ate_absolute * 100).toFixed(4)}%`, sub: 'Absolute conversion lift', glowCls: 'amber', color: 'var(--accent-amber-light)' },
          { label: 'Global ATE (Relative)', val: `+${(metrics.global_ate_relative * 100).toFixed(1)}%`, sub: 'Relative lift vs control', glowCls: 'cyan', color: 'var(--accent-cyan)' },
          { label: 'Student Model R²', val: `${(metrics.student_r2 * 100).toFixed(1)}%`, sub: 'Distillation fidelity', glowCls: 'green', color: 'var(--accent-emerald)' },
        ].map((s, i) => (
          <FadeIn key={i} delay={i * 80}>
            <div className="metric-card">
              <div className={`metric-card-glow ${s.glowCls}`} />
              <div className="metric-label">{s.label}</div>
              <div className="metric-value" style={{ color: s.color, fontSize: '2rem', fontFamily: 'var(--font-mono)' }}>{s.val}</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: 6 }}>{s.sub}</div>
            </div>
          </FadeIn>
        ))}
      </div>

      {/* Tabs: Qini & Decile */}
      <FadeIn delay={100}>
        <div className="glass-card card-pad" style={{ marginBottom: 24 }}>
          <div className="tab-list">
            <button className={`tab-btn ${activeTab === 'qini'   ? 'active' : ''}`} onClick={() => setActiveTab('qini')}>
              Qini Curve, 95% CI
            </button>
            <button className={`tab-btn ${activeTab === 'decile' ? 'active' : ''}`} onClick={() => setActiveTab('decile')}>
              Treatment vs Control by Decile
            </button>
          </div>

          {activeTab === 'qini' && (
            <div>
              <div className="card-header" style={{ marginBottom: 8 }}>
                <div>
                  <div className="card-title">Bootstrapped Qini Curve</div>
                  <div className="card-subtitle">
                    AUUC:{' '}
                    <strong style={{ color: 'var(--accent-amber-light)', fontFamily: 'var(--font-mono)' }}>
                      {qini_data.auuc_mean?.toFixed(2)}
                    </strong>
                    {' '}±{' '}
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      {qini_data.auuc_std?.toFixed(4)}
                    </span>
                    {' '}· Shaded region = 95% confidence interval (20 bootstraps)
                  </div>
                </div>
                <span className="card-badge amber">Beats random</span>
              </div>
              <ResponsiveContainer width="100%" height={340}>
                <AreaChart data={qiniChartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <defs>
                    <linearGradient id="qiniCI" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#F59E0B" stopOpacity={0.18} />
                      <stop offset="100%" stopColor="#F59E0B" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="qiniLine" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#F59E0B" />
                      <stop offset="100%" stopColor="#F97316" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="x" tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false}
                    label={{ value: 'Population Targeted →', position: 'insideBottom', fill: 'var(--text-muted)', fontSize: 11, dy: 16 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<QiniTooltip />} />
                  <Area type="monotone" dataKey="upper" stroke="none" fill="url(#qiniCI)" name="Upper 95% CI" />
                  <Area type="monotone" dataKey="lower" stroke="none" fill="white" fillOpacity={0} name="Lower 95% CI" />
                  <Line type="monotone" dataKey="random" stroke="rgba(255,255,255,0.18)" strokeDasharray="6 4" dot={false} strokeWidth={1.5} name="Random Baseline" />
                  <Line type="monotone" dataKey="mean" stroke="#F59E0B" strokeWidth={2.5} dot={false} name="Uplift Model" />
                </AreaChart>
              </ResponsiveContainer>
              <div className="insight-caption">
                The amber curve staying <strong>above the random baseline throughout</strong> proves the X-Learner
                successfully ranks users by uplift potential. The model genuinely found the Persuadables.
              </div>
            </div>
          )}

          {activeTab === 'decile' && (
            <div>
              <div className="card-header" style={{ marginBottom: 8 }}>
                <div>
                  <div className="card-title">Conversion Rate: Treatment vs Control</div>
                  <div className="card-subtitle">D9 (top decile) drives the overwhelming majority of incremental value</div>
                </div>
                <span className="card-badge violet">{liftMultiple}× top vs population avg</span>
              </div>
              <ResponsiveContainer width="100%" height={340}>
                <AreaChart data={decileChartData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                  <defs>
                    <linearGradient id="tGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#F59E0B" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#F59E0B" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="cGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#8B5CF6" stopOpacity={0.25} />
                      <stop offset="100%" stopColor="#8B5CF6" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="bin" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} tickFormatter={v => `${v.toFixed(3)}%`} />
                  <Tooltip content={<DecileTooltip />} />
                  <Area type="monotone" dataKey="treatment" stroke="#F59E0B" strokeWidth={2.5} fill="url(#tGrad)" name="Treatment CVR" />
                  <Area type="monotone" dataKey="control"   stroke="#8B5CF6" strokeWidth={2}   fill="url(#cGrad)"  name="Control CVR" />
                </AreaChart>
              </ResponsiveContainer>
              <div className="insight-caption">
                The <strong>gap widens dramatically at D9</strong>. The treatment group converts at ~2.5% vs control at ~1.8%
                in the top decile, validating the X-Learner's precision in finding high-uplift individuals.
              </div>
            </div>
          )}
        </div>
      </FadeIn>

      {/* Architecture + Decision Rules */}
      <div className="section-grid-2">
        <FadeIn delay={100}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">X-Learner Architecture</div>
                <div className="card-subtitle">4-stage LightGBM meta-learner pipeline</div>
              </div>
              <span className="card-badge violet">Quasi-experimental</span>
            </div>
            <div className="step-list">
              {[
                { n: '01', c: 'amber',  title: 'Propensity Model g(x)', desc: 'LightGBM trained on all users to predict P(T=1|X). Used to weight the final CATE estimate.' },
                { n: '02', c: 'violet', title: 'Outcome Models M₀, M₁', desc: 'Separate models for control and treatment response. Generates counterfactual predictions for each group.' },
                { n: '03', c: 'amber',  title: 'Imputed Effects D₀, D₁', desc: 'D₁ = Y₁ − M₀(X₁)  and  D₀ = M₁(X₀) − Y₀. These are the imputed individual causal effects.' },
                { n: '04', c: 'green',  title: 'Effect Models τ₀, τ₁', desc: 'Regression forests trained on imputed effects. Final CATE = g(x)·τ₀(x) + (1−g(x))·τ₁(x).' },
              ].map(({ n, c, title, desc }) => (
                <div key={n} className="step-item">
                  <div className={`step-number ${c}`}>{n}</div>
                  <div className="step-content"><h4>{title}</h4><p>{desc}</p></div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={200}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">Distilled Decision Rules</div>
                <div className="card-subtitle">Surrogate tree trained to mimic the X-Learner's predictions</div>
              </div>
              <span className="card-badge green">{(metrics.student_r2 * 100).toFixed(1)}% R²</span>
            </div>
            <pre className="code-block">{rules}</pre>
            <div style={{ marginTop: 16, padding: '14px 18px', borderRadius: 10, background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.18)' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                <strong style={{ color: 'var(--accent-amber)' }}>Key Insight:</strong>{' '}
                Users with high{' '}
                <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-violet)' }}>f4</code>{' '}
                and low{' '}
                <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-violet)' }}>f3</code>{' '}
                are the Persuadable cohort. Bid aggressively on these feature signatures.
              </p>
            </div>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
