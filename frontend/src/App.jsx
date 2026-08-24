import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ExecutiveSummary from './pages/ExecutiveSummary';
import CausalAnalysis from './pages/CausalAnalysis';
import BanditSimulation from './pages/BanditSimulation';
import InferenceEngine from './pages/InferenceEngine';

export default function App() {
  const [page, setPage] = useState('executive');
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/insights.json')
      .then(r => r.json())
      .then(d => { setInsights(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const pageComponent = () => {
    if (loading) return <LoadingState />;
    if (!insights) return <ErrorState />;
    switch (page) {
      case 'executive': return <ExecutiveSummary insights={insights} />;
      case 'causal':    return <CausalAnalysis insights={insights} />;
      case 'bandit':    return <BanditSimulation insights={insights} />;
      case 'inference': return <InferenceEngine />;
      default:          return <ExecutiveSummary insights={insights} />;
    }
  };

  return (
    <>
      <div className="bg-mesh" aria-hidden="true" />
      <div className="app-shell">
        <Sidebar activePage={page} onNavigate={setPage} />
        <main className="main-content">
          {pageComponent()}
        </main>
      </div>
    </>
  );
}

function LoadingState() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '80vh', flexDirection: 'column', gap: 16 }}>
      <div style={{ width: 48, height: 48, borderRadius: '50%', border: '3px solid rgba(59,130,246,0.2)', borderTopColor: 'var(--accent-blue)', animation: 'spin 1s linear infinite' }} />
      <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>Loading pipeline results...</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function ErrorState() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '80vh', flexDirection: 'column', gap: 16 }}>
      <p style={{ color: 'var(--accent-red)', fontSize: '1rem', fontWeight: 600 }}>Failed to load insights.json</p>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Run <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>python pipeline.py</code> to generate the data file.</p>
    </div>
  );
}
