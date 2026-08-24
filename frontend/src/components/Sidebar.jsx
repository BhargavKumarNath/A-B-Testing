import React from 'react';
import { LayoutDashboard, FlaskConical, TrendingUp, Zap, GitBranch, ExternalLink } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'executive', label: 'Executive Summary', icon: LayoutDashboard, badge: '01' },
  { id: 'causal',    label: 'Causal Analysis',   icon: FlaskConical,     badge: '02' },
  { id: 'bandit',    label: 'Bandit Simulation',  icon: TrendingUp,       badge: '03' },
  { id: 'inference', label: 'Inference Engine',   icon: Zap,              badge: '04' },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">
          {/* Profit chart icon, thematic for the dashboard */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0A0D14" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>
            <polyline points="16 7 22 7 22 13"/>
          </svg>
        </div>
        <div className="logo-text">
          <h2>ABlytics</h2>
          <p>Causal AI · War Room</p>
        </div>
      </div>

      <p className="nav-section-label">Analysis Phases</p>
      <nav>
        {NAV_ITEMS.map(({ id, label, icon: Icon, badge }) => (
          <div
            key={id}
            className={`nav-item ${activePage === id ? 'active' : ''}`}
            onClick={() => onNavigate(id)}
          >
            <Icon size={15} />
            {label}
            <span className="nav-badge">{badge}</span>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div style={{ display: 'flex', gap: 14, marginBottom: 12, flexWrap: 'wrap' }}>
          <a
            href="https://github.com/BhargavKumarNath/A-B-Testing"
            target="_blank"
            rel="noreferrer"
            style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: '0.78rem', textDecoration: 'none', transition: 'color 0.15s' }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--accent-amber)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <GitBranch size={13} /> Source
          </a>
          <a
            href="https://ablytics.streamlit.app"
            target="_blank"
            rel="noreferrer"
            style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: '0.78rem', textDecoration: 'none', transition: 'color 0.15s' }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--accent-amber)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <ExternalLink size={13} /> Live Demo
          </a>
        </div>
        <p>Pipeline v2.0 · 14M rows<br />Criteo Uplift Dataset</p>
      </div>
    </aside>
  );
}
