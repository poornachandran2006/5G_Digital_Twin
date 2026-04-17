import { useTheme } from '../../context/ThemeContext';

const NAV = [
  { id: 'overview',    label: 'Overview',     icon: '◉', desc: 'System health' },
  { id: 'map',         label: 'Network Map',   icon: '⬡', desc: 'Live UE positions' },
  { id: 'kpi',         label: 'KPIs',          icon: '△', desc: 'Per-cell metrics' },
  { id: 'predictions', label: 'Predictions',   icon: '◈', desc: 'Congestion forecast' },
  { id: 'shap',        label: 'SHAP',          icon: '⬢', desc: 'Model explainability' },
  { id: 'anomaly',     label: 'Anomaly',       icon: '⚠', desc: 'Anomaly detection' },
  { id: 'abtest',      label: 'A/B Test',      icon: '⚖', desc: 'PPO vs rule-based' },
  { id: 'rl',          label: 'RL Agent',      icon: '⬟', desc: 'PPO decisions' },
];

export default function Sidebar({ active, onNavigate }) {
  const { theme } = useTheme();

  return (
    <aside
      className="w-52 flex flex-col shrink-0 py-5"
      style={{
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border)',
        transition: 'var(--transition)',
      }}
    >
      {/* Logo area */}
      <div className="px-5 mb-7">
        <div className="flex items-center gap-2 mb-1">
          <div
            className="w-6 h-6 rounded flex items-center justify-center text-xs font-bold"
            style={{ background: 'var(--accent-dim)', color: 'var(--accent)' }}
          >
            5G
          </div>
          <span
            className="text-xs font-semibold tracking-wider uppercase"
            style={{ color: 'var(--text-primary)' }}
          >
            Digital Twin
          </span>
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: '1px', background: 'var(--border)', margin: '0 20px 12px' }} />

      {/* Nav label */}
      <p
        className="px-5 mb-2"
        style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.15em', textTransform: 'uppercase', fontFamily: 'monospace' }}
      >
        Panels
      </p>

      {/* Nav items */}
      <nav className="flex flex-col gap-0.5 px-2">
        {NAV.map((n) => {
          const isActive = active === n.id;
          return (
            <button
              key={n.id}
              type="button"
              onClick={() => onNavigate(n.id)}
              className="flex items-start gap-3 px-3 py-2.5 rounded-lg text-left w-full transition-all duration-150"
              style={{
                background: isActive ? 'var(--accent-dim)' : 'transparent',
                borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
              }}
            >
              <span style={{ fontSize: '13px', marginTop: '1px', flexShrink: 0 }}>{n.icon}</span>
              <div>
                <div style={{ fontSize: '12px', fontWeight: isActive ? 600 : 400, lineHeight: 1.3 }}>
                  {n.label}
                </div>
                <div style={{ fontSize: '9px', color: isActive ? 'var(--accent)' : 'var(--text-muted)', opacity: 0.85, fontFamily: 'monospace', marginTop: '1px' }}>
                  {n.desc}
                </div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Bottom — theme indicator */}
      <div className="mt-auto px-5 pt-4">
        <div
          style={{
            fontSize: '9px',
            color: 'var(--text-muted)',
            fontFamily: 'monospace',
            letterSpacing: '0.08em',
            borderTop: '1px solid var(--border)',
            paddingTop: '12px',
          }}
        >
          <div style={{ marginTop: '3px', color: theme === 'dark' ? 'var(--accent)' : 'var(--purple)' }}>
            {theme === 'dark' ? '◐ Dark mode' : '◑ Light mode'}
          </div>
        </div>
      </div>
    </aside>
  );
}