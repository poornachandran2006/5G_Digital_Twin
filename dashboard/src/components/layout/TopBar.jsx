import { useSim } from '../../context/SimContext';
import { useTheme } from '../../context/ThemeContext';

export default function TopBar({ onMenuClick }) {
  const { state } = useSim();
  const { theme, toggle } = useTheme();

  const tick = state?.currentTick?.tick ?? null;
  const totalThroughput =
    state?.currentTick?.kpis?.total_throughput?.toFixed(0) ?? null;
  const cells = state?.currentTick?.cells ?? [];
  const activeUEs = state?.currentTick?.kpis?.active_ues ?? 0;

  const maxLoad =
    cells.length > 0
      ? Math.max(...cells.map((c) => c.load_percent ?? 0))
      : 0;

  const networkStatus =
    maxLoad >= 0.9 ? 'CRITICAL' : maxLoad >= 0.7 ? 'WARNING' : 'HEALTHY';

  const statusColor =
    networkStatus === 'CRITICAL'
      ? 'var(--red)'
      : networkStatus === 'WARNING'
      ? 'var(--amber)'
      : 'var(--green)';

  return (
    <header
      className="
        flex flex-col sm:flex-row sm:items-center sm:justify-between
        gap-2 sm:gap-0 px-3 sm:px-5 py-2 sm:py-0 shrink-0
      "
      style={{
        minHeight: '52px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
        transition: 'var(--transition)',
      }}
    >
      {/* TOP ROW (mobile layout) */}
      <div className="flex items-center justify-between sm:hidden">
        
        {/* Hamburger */}
        <button
          onClick={onMenuClick}
          className="flex items-center justify-center w-9 h-9 rounded-md"
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
          }}
        >
          <div className="flex flex-col gap-[3px]">
            <span className="block w-4 h-[2px] bg-[var(--text-primary)]"></span>
            <span className="block w-4 h-[2px] bg-[var(--text-primary)]"></span>
            <span className="block w-4 h-[2px] bg-[var(--text-primary)]"></span>
          </div>
        </button>

        {/* Title (mobile) */}
        <span className="text-[12px] font-semibold text-[var(--text-primary)]">
          5G Digital Twin
        </span>

        {/* Theme toggle (mobile compact) */}
        <button
          onClick={toggle}
          className="flex items-center justify-center w-9 h-9 rounded-md"
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
          }}
        >
          <span className="text-[13px]">
            {theme === 'dark' ? '☀' : '☽'}
          </span>
        </button>
      </div>

      {/* DESKTOP + TABLET ROW */}
      <div className="hidden sm:flex w-full items-center justify-between">
        
        {/* LEFT */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: tick ? 'var(--green)' : 'var(--text-muted)',
                boxShadow: tick ? '0 0 6px var(--green)' : 'none',
                animation: tick ? 'pulse 2s infinite' : 'none',
              }}
            />
            <span className="font-mono text-[11px] text-[var(--text-muted)]">
              TICK
            </span>
            <span className="font-mono text-[13px] font-semibold text-[var(--accent)]">
              {tick ?? '—'}
            </span>
          </div>

          <div className="w-[1px] h-[20px] bg-[var(--border)]" />

          <span className="font-mono text-[11px] text-[var(--text-muted)]">
            <span className="text-[var(--text-secondary)]">
              {totalThroughput ?? '—'}
            </span>
            <span className="ml-[3px]">Mbps</span>
          </span>
        </div>

        {/* CENTER */}
        <div className="flex items-center gap-3">
          <span className="text-[12px] font-semibold text-[var(--text-primary)] tracking-wide">
            5G Network Digital Twin
          </span>
        </div>

        {/* RIGHT */}
        <div className="flex items-center gap-3">
          <div
            className="flex items-center gap-1.5 px-3 py-1 rounded-md"
            style={{
              fontSize: '10px',
              fontFamily: 'monospace',
              color: 'var(--text-muted)',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background:
                  cells.length === 3 ? 'var(--green)' : 'var(--red)',
              }}
            />
            <span>gNB {cells.length}/3</span>
            <span className="mx-1 text-[var(--border-accent)]">·</span>
            <span>UE {activeUEs}/20</span>
          </div>

          <button
            onClick={toggle}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg"
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-accent)',
              color: 'var(--text-secondary)',
              fontSize: '11px',
              fontFamily: 'monospace',
            }}
          >
            <span>{theme === 'dark' ? '☀' : '☽'}</span>
            <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}