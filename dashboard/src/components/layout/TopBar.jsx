import { useSim } from '../../context/SimContext';
import { useTheme } from '../../context/ThemeContext';

export default function TopBar() {
  const { state } = useSim();
  const { theme, toggle } = useTheme();
  const tick = state?.currentTick?.tick ?? '—';
  const totalThroughput = state?.currentTick?.kpis?.total_throughput?.toFixed(0) ?? '—';

  return (
    <header
      className="flex items-center justify-between px-5 shrink-0"
      style={{
        height: '52px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
        transition: 'var(--transition)',
      }}
    >
      {/* Left — tick counter */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="live-dot" />
          <span style={{ fontFamily: 'monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
            TICK
          </span>
          <span style={{ fontFamily: 'monospace', fontSize: '13px', fontWeight: 600, color: 'var(--accent)' }}>
            {tick}
          </span>
        </div>
        <div
          style={{
            width: '1px', height: '20px',
            background: 'var(--border)',
          }}
        />
        <div style={{ fontFamily: 'monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
          <span style={{ color: 'var(--text-secondary)' }}>{totalThroughput}</span>
          <span style={{ marginLeft: '3px' }}>Mbps total</span>
        </div>
      </div>

      {/* Center — title */}
      <div className="flex items-center gap-2">
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>
          5G Network Digital Twin
        </span>
        <span
          style={{
            fontSize: '9px',
            padding: '2px 7px',
            borderRadius: '99px',
            background: 'var(--accent-dim)',
            color: 'var(--accent)',
            fontFamily: 'monospace',
            border: '1px solid var(--accent-glow)',
          }}
        >
          LIVE
        </span>
      </div>

      {/* Right — theme toggle + status */}
      <div className="flex items-center gap-3">
        <div
          style={{
            fontSize: '10px',
            fontFamily: 'monospace',
            color: 'var(--text-muted)',
            padding: '3px 8px',
            borderRadius: '6px',
            background: 'var(--accent-dim)',
            border: '1px solid var(--border)',
          }}
        >
          <span style={{ marginRight: '4px' }}>Status</span>
        </div>

        {/* Dark / light toggle */}
        <button
          onClick={toggle}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all duration-200"
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-accent)',
            color: 'var(--text-secondary)',
            fontSize: '11px',
            fontFamily: 'monospace',
            cursor: 'pointer',
          }}
        >
          <span style={{ fontSize: '14px' }}>{theme === 'dark' ? '☀' : '☽'}</span>
          <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </button>
      </div>
    </header>
  );
}