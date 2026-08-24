import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, Zap, DollarSign } from 'lucide-react';
import { AnimatedNumber, FadeIn } from '../components/Animations';

/* ── Sparkline micro-chart (pure SVG, no deps) ── */
function Sparkline({ data, color, width = 80, height = 32 }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const pathD = `M ${pts.join(' L ')}`;
  const fillD = `M ${pts[0]} L ${pts.join(' L ')} L ${width},${height} L 0,${height} Z`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="kpi-sparkline">
      <defs>
        <linearGradient id={`sg-${color.replace('#','')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#sg-${color.replace('#','')})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const ProfitTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const v = payload[0].value;
  return (
    <div style={{ background: 'rgba(10,13,20,0.97)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '10px 14px', backdropFilter: 'blur(16px)' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginBottom: 4 }}>Strategy</p>
      <p style={{ color: v >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>
        {v >= 0 ? '+' : ''}${v.toFixed(4)}/user
      </p>
    </div>
  );
};

const DecileTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const v = payload[0].value;
  return (
    <div style={{ background: 'rgba(10,13,20,0.97)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '10px 14px' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginBottom: 4 }}>Decile {label}</p>
      <p style={{ color: v >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
        {(v * 100).toFixed(4)}% lift
      </p>
    </div>
  );
};

/* Sparkline data (illustrative trajectories) */
const SPARKLINES = {
  lift:    [0.2, 0.35, 0.45, 0.42, 0.5, 0.55, 0.594],
  loss:    [0, -0.02, -0.04, -0.055, -0.065, -0.069],
  profit:  [0, 0.01, 0.03, 0.055, 0.07, 0.083],
  latency: [5, 3, 2, 1.5, 1.2, 1, 0.8],
};

const ARCHETYPE_DATA = [
  { cls: 'persuadable',  icon: '🎯', name: 'Persuadables',  pct: '0.85%', desc: 'High uplift, moderate baseline. The core profit segment. Bid aggressively.', color: 'var(--accent-emerald)' },
  { cls: 'sure-thing',  icon: '✅', name: 'Sure Things',   pct: '9.87%', desc: 'Convert regardless of the ad. Bidding here wastes budget.', color: 'var(--accent-amber)' },
  { cls: 'lost-cause',  icon: '❌', name: 'Lost Causes',   pct: '89.22%', desc: 'Low conversion probability regardless of treatment. Skip.', color: 'var(--accent-rose)' },
  { cls: 'sleeping-dog', icon: '🌙', name: 'Sleeping Dogs', pct: '0.06%', desc: 'Ads actively decrease conversion probability. Never bid.', color: 'var(--accent-violet)' },
];

export default function ExecutiveSummary({ insights }) {
  const { metrics, decile_stats } = insights;
  const improvement = metrics.bandit_profit_per_user - metrics.baseline_profit_per_user;

  const profitData = [
    { name: 'A/B Baseline', value: metrics.baseline_profit_per_user },
    { name: 'Causal Bandit', value: metrics.bandit_profit_per_user },
  ];

  const decileData = [...decile_stats]
    .sort((a, b) => Number(a.bin) - Number(b.bin))
    .map(d => ({ bin: d.bin, lift: d.actual_lift, predicted: d.mean_pred_uplift }));

  const topDecile = decileData.find(d => d.bin === '9');
  const avgLift = metrics.global_ate_absolute;
  const liftMultiple = topDecile ? (topDecile.lift / (avgLift || 1)).toFixed(1) : '?';

  const KPI_CARDS = [
    {
      icon: TrendingUp, iconCls: 'amber', glowCls: 'amber',
      label: 'A/B Lift (Conversion)',
      number: <><AnimatedNumber value={metrics.global_ate_relative * 100} decimals={1} />%</>,
      numCls: 'amber',
      pill: { label: 'Statistically significant', type: 'amber' },
      sparkData: SPARKLINES.lift, sparkColor: '#F59E0B',
    },
    {
      icon: TrendingDown, iconCls: 'rose', glowCls: 'rose',
      label: 'Baseline Profit / User',
      number: <><span style={{ fontSize: '2rem' }}>-$</span><AnimatedNumber value={Math.abs(metrics.baseline_profit_per_user)} decimals={4} /></>,
      numCls: 'rose',
      pill: { label: '↓ Economic Loss', type: 'down', dot: true },
      sparkData: SPARKLINES.loss, sparkColor: '#F43F5E',
    },
    {
      icon: DollarSign, iconCls: 'emerald', glowCls: 'emerald',
      label: 'Bandit Profit / User',
      number: <><span style={{ fontSize: '2rem' }}>+$</span><AnimatedNumber value={metrics.bandit_profit_per_user} decimals={4} /></>,
      numCls: 'emerald',
      pill: { label: '↑ Profitable', type: 'up' },
      sparkData: SPARKLINES.profit, sparkColor: '#10B981',
    },
    {
      icon: Zap, iconCls: 'violet', glowCls: 'violet',
      label: 'Inference Latency',
      number: '<1ms',
      numCls: 'violet',
      pill: { label: 'RTB-ready', type: 'neutral' },
      sparkData: SPARKLINES.latency, sparkColor: '#8B5CF6',
    },
  ];

  return (
    <div>
      {/* Page Header */}
      <FadeIn>
        <div className="page-header">
          <div className="page-eyebrow"><span>▸</span> Phase 1 of 4 · Executive Summary</div>
          <h1>
            From <span className="gradient-text-loss">Loss</span> to{' '}
            <span className="gradient-text-profit">Profit</span>
          </h1>
          <p>
            Standard A/B testing revealed +59.4% conversion lift, yet the campaign was
            bleeding <strong style={{ color: 'var(--accent-rose)' }}>$0.07 per user</strong>.
            By layering X-Learner causal models onto a profit-aware LinUCB bandit,
            we reversed the economics across 14M impressions.
          </p>
        </div>
      </FadeIn>

      {/* Narrative Banner */}
      <FadeIn delay={60}>
        <div className="narrative-banner">
          <div className="narrative-banner-label">📌 The Core Insight</div>
          <div className="narrative-banner-text">
            A/B lift was real, but targeting everyone was economically irrational.
            Only the top 10% of users (Persuadables) generate meaningful uplift,
            and the causal model found them, turning a{' '}
            <span style={{ color: 'var(--accent-rose)', fontWeight: 700 }}>$0.069/user loss</span> into a{' '}
            <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>$0.083/user profit</span>.
          </div>
        </div>
      </FadeIn>

      {/* KPI Playcards */}
      <div className="kpi-grid">
        {KPI_CARDS.map(({ icon: Icon, iconCls, glowCls, label, number, numCls, pill, sparkData, sparkColor }, i) => (
          <FadeIn key={i} delay={i * 80}>
            <div className="kpi-playcard">
              <div className={`kpi-playcard-glow ${glowCls}`} />
              <div className="kpi-top-row">
                <div className={`kpi-icon ${iconCls}`}><Icon size={17} /></div>
                <Sparkline data={sparkData} color={sparkColor} />
              </div>
              <div className="kpi-label">{label}</div>
              <div className={`kpi-number ${numCls}`}>{number}</div>
              <div className="kpi-bottom-row">
                <span className={`trend-pill ${pill.type}`}>
                  {pill.dot && <span className="danger-dot" />}
                  {pill.label}
                </span>
              </div>
            </div>
          </FadeIn>
        ))}
      </div>

      {/* Charts Row */}
      <div className="section-grid-2">
        {/* Profit Comparison */}
        <FadeIn delay={100}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">Unit Economics Turnaround</div>
                <div className="card-subtitle">Net profit per user: Baseline vs Causal Bandit strategy</div>
              </div>
              <span className="card-badge emerald">+${improvement.toFixed(4)} delta</span>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={profitData} barSize={64} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} tickFormatter={v => `$${v.toFixed(3)}`} />
                <Tooltip content={<ProfitTooltip />} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.18)" strokeDasharray="5 4" label={{ value: 'Break-even', position: 'insideTopRight', fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                  {profitData.map((entry, i) => (
                    <Cell key={i} fill={entry.value >= 0 ? 'url(#profitGrad)' : 'url(#lossGrad)'} />
                  ))}
                </Bar>
                <defs>
                  <linearGradient id="profitGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" stopOpacity={0.95} />
                    <stop offset="100%" stopColor="#06B6D4" stopOpacity={0.6} />
                  </linearGradient>
                  <linearGradient id="lossGrad" x1="0" y1="1" x2="0" y2="0">
                    <stop offset="0%" stopColor="#F43F5E" stopOpacity={0.95} />
                    <stop offset="100%" stopColor="#F97316" stopOpacity={0.6} />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
            <div className="insight-caption">
              The Causal Bandit flips the sign entirely. Not marginal improvement but a{' '}
              <strong>fundamental strategy switch</strong> from "treat everyone" to "treat only Persuadables."
            </div>
          </div>
        </FadeIn>

        {/* Uplift by Decile */}
        <FadeIn delay={200}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">Uplift by Predicted Decile</div>
                <div className="card-subtitle">Actual treatment effect within X-Learner predicted score buckets</div>
              </div>
              <span className="card-badge amber">{liftMultiple}× top vs avg</span>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={decileData} barSize={24} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="bin" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false}
                  label={{ value: 'Decile →', position: 'insideRight', fill: 'var(--text-muted)', fontSize: 10, dy: 14 }} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} tickFormatter={v => `${(v * 100).toFixed(2)}%`} />
                <Tooltip content={<DecileTooltip />} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
                {/* Annotation on D9 */}
                {topDecile && (
                  <ReferenceLine x="9" stroke="rgba(245,158,11,0.4)" strokeDasharray="3 3"
                    label={{ value: `D9 spike: ${(topDecile.lift*100).toFixed(3)}%`, position: 'insideTopLeft', fill: 'var(--accent-amber)', fontSize: 9 }} />
                )}
                <Bar dataKey="lift" radius={[4, 4, 0, 0]}>
                  {decileData.map((entry, i) => (
                    <Cell key={i} fill={entry.bin === '9' ? 'url(#amberGrad)' : entry.lift >= 0 ? 'url(#tealGrad)' : 'url(#roseGrad)'} />
                  ))}
                </Bar>
                <defs>
                  <linearGradient id="amberGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#F59E0B" stopOpacity={0.95} />
                    <stop offset="100%" stopColor="#F59E0B" stopOpacity={0.5} />
                  </linearGradient>
                  <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06B6D4" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#06B6D4" stopOpacity={0.3} />
                  </linearGradient>
                  <linearGradient id="roseGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#F43F5E" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#F43F5E" stopOpacity={0.3} />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
            <div className="insight-caption">
              <strong>Decile 9 alone</strong> accounts for the vast majority of campaign value.
              the model correctly concentrates predicted scores onto the Persuadable population.
            </div>
          </div>
        </FadeIn>
      </div>

      {/* Audience Archetypes + Strategic Narrative */}
      <div className="section-grid-1-2">
        <FadeIn delay={100}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">Audience Archetypes</div>
                <div className="card-subtitle">Population segmentation by causal treatment response</div>
              </div>
            </div>
            <div className="archetype-grid">
              {ARCHETYPE_DATA.map(a => (
                <div key={a.cls} className={`archetype-card ${a.cls}`}>
                  <div className="archetype-icon">{a.icon}</div>
                  <div className="archetype-name">{a.name}</div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: a.color, marginBottom: 6 }}>{a.pct}</div>
                  <div className="archetype-desc">{a.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={200}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">The Strategic Narrative</div>
                <div className="card-subtitle">Why standard lift metrics are dangerous without causal reasoning</div>
              </div>
              <span className="card-badge amber">5-Phase Pipeline</span>
            </div>
            <div className="step-list">
              {[
                { n: '01', c: 'amber',  title: 'Experiment Validation', desc: 'SRM Chi-Square test + SMD audit confirmed the 85/15 traffic split was clean and groups were covariate-balanced. The data was trustworthy.' },
                { n: '02', c: 'rose',   title: 'The Profitability Trap', desc: 'A/B test showed +59.4% lift. But (Lift × $10 LTV) = $0.001 < $0.10 cost-per-ad. Statistically significant, economically disastrous.' },
                { n: '03', c: 'violet', title: 'X-Learner CATE Estimation', desc: `Trained a 4-stage LightGBM meta-learner to predict individual treatment effects. Decile 9 showed ${liftMultiple}× the population average uplift.` },
                { n: '04', c: 'green',  title: 'Profit-Aware LinUCB Bandit', desc: 'Replayed 1M impressions with a bandit optimizing (Uplift × LTV) > Cost. Policy learned to skip Lost Causes, turning the campaign profitable.' },
                { n: '05', c: 'amber',  title: 'Knowledge Distillation', desc: `Complex X-Learner ensemble compressed to a depth-5 Decision Tree with ${(metrics.student_r2 * 100).toFixed(1)}% R² fidelity. Sub-millisecond RTB-ready inference.` },
              ].map(({ n, c, title, desc }) => (
                <div key={n} className="step-item">
                  <div className={`step-number ${c}`}>{n}</div>
                  <div className="step-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
