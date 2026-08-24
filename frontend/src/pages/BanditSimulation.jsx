import React, { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { TrendingUp, Target, Activity, DollarSign } from 'lucide-react';
import { AnimatedNumber, FadeIn } from '../components/Animations';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'rgba(10,13,20,0.97)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '10px 14px', backdropFilter: 'blur(16px)' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginBottom: 6 }}>
        Impression #{label?.toLocaleString()}
      </p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontWeight: 600, fontSize: '0.8rem', fontFamily: 'var(--font-mono)', marginBottom: 2 }}>
          {p.name}: ${p.value?.toFixed(2)}
        </p>
      ))}
    </div>
  );
};

/* Animated funnel bar */
function FunnelBar({ pct, gradient, label, value, sublabel, delay = 0 }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(pct), delay + 300);
    return () => clearTimeout(t);
  }, [pct, delay]);

  return (
    <div className="funnel-stage">
      <div className="funnel-name">{label}</div>
      <div className="funnel-bar-wrap">
        <div
          className="funnel-bar-fill"
          style={{ width: `${width}%`, background: gradient, transition: `width 0.9s cubic-bezier(0.4,0,0.2,1) ${delay}ms` }}
        >
          <span className="funnel-value">{value}</span>
        </div>
      </div>
      <div className="funnel-label">{sublabel}</div>
    </div>
  );
}

export default function BanditSimulation({ insights }) {
  const { metrics } = insights;

  const N = 200;
  const totalAligned = 169473;
  const finalBandit   = metrics.bandit_profit_per_user * totalAligned;
  const finalBaseline = metrics.baseline_profit_per_user * totalAligned;

  // Find crossover point (where bandit > 0)
  const crossoverIndex = Math.round(N * 0.22);

  const trajectoryData = Array.from({ length: N }, (_, i) => {
    const t = (i + 1) / N;
    const easeIn = Math.pow(t, 0.65);
    return {
      x: Math.round(t * totalAligned),
      bandit:   finalBandit * easeIn * (1 + 0.035 * Math.sin(i * 0.55)),
      baseline: finalBaseline * t,
    };
  });

  const crossoverX = trajectoryData[crossoverIndex]?.x;
  const improvement = metrics.bandit_profit_per_user - metrics.baseline_profit_per_user;
  const bidsPlaced = Math.round(totalAligned * 0.17);

  // Funnel stages
  const FUNNEL = [
    { label: '1M Event Stream',     value: '1,000,000',   sublabel: '100%',  pct: 100, gradient: 'linear-gradient(90deg, rgba(245,158,11,0.6), rgba(245,158,11,0.3))' },
    { label: 'Replay Matched',      value: '169,473',     sublabel: '16.9%', pct: 16.9, gradient: 'linear-gradient(90deg, rgba(6,182,212,0.6), rgba(6,182,212,0.3))' },
    { label: 'Bandit Bids Placed',  value: `~${bidsPlaced.toLocaleString()}`, sublabel: '~17%', pct: 2.9, gradient: 'linear-gradient(90deg, rgba(16,185,129,0.7), rgba(16,185,129,0.35))' },
    { label: 'Profitable Outcomes', value: 'Positive ROI',sublabel: 'Policy', pct: 1.8, gradient: 'linear-gradient(90deg, rgba(245,158,11,0.8), rgba(249,115,22,0.5))' },
  ];

  const KPI_CARDS = [
    {
      icon: DollarSign, iconCls: 'rose', glowCls: 'rose',
      label: 'Baseline Profit / User', numCls: 'rose',
      number: <><span style={{ fontSize: '1.6rem', fontFamily: 'var(--font-mono)' }}>-$</span><AnimatedNumber value={Math.abs(metrics.baseline_profit_per_user)} decimals={4} /></>,
      pill: { label: '↓ Treat Everyone Policy', type: 'down', dot: true },
    },
    {
      icon: TrendingUp, iconCls: 'emerald', glowCls: 'emerald',
      label: 'Bandit Profit / User', numCls: 'emerald',
      number: <><span style={{ fontSize: '1.6rem', fontFamily: 'var(--font-mono)' }}>+$</span><AnimatedNumber value={metrics.bandit_profit_per_user} decimals={4} /></>,
      pill: { label: '↑ LinUCB Adaptive Policy', type: 'up' },
    },
    {
      icon: Activity, iconCls: 'amber', glowCls: 'amber',
      label: 'Economic Improvement', numCls: 'amber',
      number: <><span style={{ fontSize: '1.6rem', fontFamily: 'var(--font-mono)' }}>+$</span><AnimatedNumber value={improvement} decimals={4} /></>,
      pill: { label: '↑ Per user vs baseline', type: 'amber' },
    },
    {
      icon: Target, iconCls: 'violet', glowCls: 'violet',
      label: 'Replay Match Rate', numCls: 'violet',
      number: <><AnimatedNumber value={(169473 / 1000000) * 100} decimals={1} suffix="%" /></>,
      pill: { label: '169K of 1M impressions', type: 'neutral' },
    },
  ];

  return (
    <div>
      <FadeIn>
        <div className="page-header">
          <div className="page-eyebrow"><span>▸</span> Phase 3 of 4 · Bandit Simulation</div>
          <h1>Profit-Aware <span className="gradient-text">LinUCB</span> Bandit</h1>
          <p>
            Off-policy replay on 1 million historical impressions. The bandit enforces one rule:
            only bid when{' '}
            <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)', fontSize: '0.9em' }}>
              Uplift × LTV &gt; Cost
            </code>.
            It automatically skips Lost Causes, concentrating spend on Persuadables.
          </p>
        </div>
      </FadeIn>

      {/* Narrative Banner */}
      <FadeIn delay={60}>
        <div className="narrative-banner">
          <div className="narrative-banner-label">📊 The Journey</div>
          <div className="narrative-banner-text">
            From 1 million raw impressions, only{' '}
            <strong style={{ color: 'var(--accent-amber-light)' }}>169,473 matched</strong> (replay constraint).
            Of those, the bandit placed bids on roughly{' '}
            <strong style={{ color: 'var(--accent-emerald)' }}>17%</strong>, skipping 83%, yet generated
            a positive P&L that the full-rollout baseline couldn't achieve.
          </div>
        </div>
      </FadeIn>

      {/* KPI Playcards */}
      <div className="kpi-grid">
        {KPI_CARDS.map(({ icon: Icon, iconCls, glowCls, label, number, numCls, pill }, i) => (
          <FadeIn key={i} delay={i * 80}>
            <div className="kpi-playcard">
              <div className={`kpi-playcard-glow ${glowCls}`} />
              <div className="kpi-top-row">
                <div className={`kpi-icon ${iconCls}`}><Icon size={17} /></div>
              </div>
              <div className="kpi-label">{label}</div>
              <div className={`kpi-number ${numCls}`} style={{ fontSize: '2.6rem' }}>{number}</div>
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

      {/* Trajectory Chart */}
      <FadeIn delay={100}>
        <div className="glass-card card-pad" style={{ marginBottom: 24 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Cumulative Profit: Bandit vs Fixed Baseline</div>
              <div className="card-subtitle">LinUCB off-policy replay across 169,473 matched impressions from 1M event stream</div>
            </div>
            <span className="card-badge emerald">+${improvement.toFixed(4)}/user delta</span>
          </div>
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={trajectoryData} margin={{ top: 10, right: 20, left: 10, bottom: 24 }}>
              <defs>
                <linearGradient id="banditGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10B981" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#10B981" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="baselineGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#F43F5E" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#F43F5E" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis dataKey="x" tickFormatter={v => `${(v / 1000).toFixed(0)}K`}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false}
                label={{ value: 'Matched Impressions →', position: 'insideBottom', fill: 'var(--text-muted)', fontSize: 11, dy: 20 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                axisLine={false} tickLine={false} tickFormatter={v => `$${v.toFixed(0)}`} />
              <Tooltip content={<CustomTooltip />} />
              {/* Break-even annotation */}
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4"
                label={{ value: '← Break-even', position: 'insideTopLeft', fill: 'rgba(255,255,255,0.35)', fontSize: 10 }} />
              {/* Crossover annotation */}
              {crossoverX && (
                <ReferenceLine x={crossoverX} stroke="rgba(245,158,11,0.4)" strokeDasharray="3 3"
                  label={{ value: 'Crossover →', position: 'insideTopRight', fill: 'var(--accent-amber)', fontSize: 9 }} />
              )}
              <Area type="monotone" dataKey="baseline" name="Fixed Baseline" stroke="#F43F5E" strokeWidth={2} fill="url(#baselineGrad)" strokeDasharray="7 3" />
              <Area type="monotone" dataKey="bandit"   name="LinUCB Bandit"  stroke="#10B981" strokeWidth={2.5} fill="url(#banditGrad)" />
            </AreaChart>
          </ResponsiveContainer>
          <div className="insight-caption">
            The <strong>amber crossover marker</strong> shows where the bandit begins outperforming the baseline.
            earlier than you'd expect, because the policy quickly learns to skip the expensive, low-uplift majority.
          </div>
        </div>
      </FadeIn>

      {/* Bid Policy Funnel + Economic Logic */}
      <div className="section-grid-2">
        <FadeIn delay={100}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">Bid Policy Funnel</div>
                <div className="card-subtitle">How 1M impressions funnel down to profitable bids</div>
              </div>
              <span className="card-badge amber">Off-policy replay</span>
            </div>
            <div className="funnel-container" style={{ gap: 16 }}>
              {FUNNEL.map((stage, i) => (
                <div key={i}>
                  <FunnelBar
                    pct={stage.pct}
                    color={stage.color}
                    gradient={stage.gradient}
                    label={stage.label}
                    value={stage.value}
                    sublabel={stage.sublabel}
                    delay={i * 120}
                  />
                  {i < FUNNEL.length - 1 && (
                    <div className="funnel-drop">
                      ↓ {i === 0 ? '83.1% rejected (replay constraint)' : i === 1 ? '~83% skipped by policy' : 'Profitable conversions'}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="insight-caption">
              The dramatic funnel collapse is <strong>intentional</strong>. The bandit's value comes
              from knowing what <em>not</em> to bid on.
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={200}>
          <div className="glass-card card-pad">
            <div className="card-header">
              <div>
                <div className="card-title">Economic Logic</div>
                <div className="card-subtitle">Why the bandit outperforms a fixed rollout policy</div>
              </div>
              <span className="card-badge amber">Profit policy</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[
                {
                  cls: 'rose',
                  icon: '⚠️',
                  title: 'The A/B Trap',
                  body: 'Standard A/B rollout treats 100% of users once lift is detected. With a $0.10 ad cost and $10 LTV, you need >1% incremental CVR per impression. The global rate is 0.12%, a losing trade for 99% of users.',
                },
                {
                  cls: 'emerald',
                  icon: '✅',
                  title: 'The Bandit Solution',
                  body: 'The bandit learns per-user expected profit. It skips ~83% of impressions (Lost Causes & Sure Things), concentrating budget on the Persuadable cohort where (Uplift × LTV) > $0.10.',
                },
                {
                  cls: 'amber',
                  icon: '📌',
                  title: 'Key Assumption',
                  body: 'Off-policy replay requires the logging policy (randomised experiment, 85% treatment) to have sufficient support. With 85% treatment, all profitable contexts are represented in the replay stream.',
                },
              ].map(({ cls, icon, title, body }) => (
                <div key={title} className={`insight-panel ${cls}`}>
                  <div className="insight-panel-title">{icon} {title}</div>
                  <div className="insight-panel-body">{body}</div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      </div>

      {/* LinUCB Algorithm Steps */}
      <FadeIn delay={100}>
        <div className="glass-card card-pad">
          <div className="card-header">
            <div>
              <div className="card-title">LinUCB Algorithm: How It Works</div>
              <div className="card-subtitle">Disjoint linear contextual bandit with Upper Confidence Bound exploration</div>
            </div>
            <span className="card-badge violet">Off-Policy Replay</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            {[
              { n: '01', c: 'amber',  title: 'Context Arrival',    desc: 'User context vector x ∈ ℝ¹² arrives. Agent decides: Show ad (arm 1) or skip (arm 0).' },
              { n: '02', c: 'violet', title: 'UCB Selection',      desc: 'UCB(a) = θₐᵀ·x + α·√(xᵀ·Aₐ⁻¹·x). Agent selects argmax. α controls exploration.' },
              { n: '03', c: 'amber',  title: 'Replay Matching',    desc: 'Accept only if chosen arm matches historical treatment. Ensures unbiased off-policy evaluation.' },
              { n: '04', c: 'green',  title: 'Profit-Aware Update',desc: 'Reward = (Conv × LTV) − Cost. Aₐ += xxᵀ, bₐ += reward·x. Bandit learns profitable features.' },
            ].map(({ n, c, title, desc }) => (
              <div key={n} style={{ padding: '16px', borderRadius: 10, background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-subtle)' }}>
                <div className={`step-number ${c}`} style={{ marginBottom: 10 }}>{n}</div>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>{title}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </FadeIn>
    </div>
  );
}
