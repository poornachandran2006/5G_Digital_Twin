export default function StatCard({ label, value, unit, trend, thresholdWarn, thresholdCrit, highlight }) {
  const num = parseFloat(value);

  let valueColor = 'var(--text-primary)';
  if (thresholdCrit !== undefined && num >= thresholdCrit) valueColor = 'var(--red)';
  else if (thresholdWarn !== undefined && num >= thresholdWarn) valueColor = 'var(--amber)';
  else if (highlight) valueColor = 'var(--accent)';

  const trendUp = trend !== undefined && trend > 0;
  const trendDown = trend !== undefined && trend < 0;

  return (
    <div
      className="rounded-xl p-4 border transition-all duration-200"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
        borderLeft: highlight ? '2px solid var(--accent)' : undefined,
      }}
    >
      <p
        className="text-xs uppercase tracking-wider mb-2 truncate"
        style={{ color: 'var(--text-muted)', fontFamily: 'monospace', fontSize: '9px', letterSpacing: '0.12em' }}
      >
        {label}
      </p>

      <div className="flex items-end gap-1.5">
        <span
          className="font-mono font-bold leading-none"
          style={{ fontSize: '22px', color: valueColor, transition: 'color 0.3s' }}
        >
          {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(1)) : (value ?? '—')}
        </span>

        {unit && (
          <span
            className="mb-0.5"
            style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace' }}
          >
            {unit}
          </span>
        )}
      </div>

      {trend !== undefined && (
        <div className="flex items-center gap-1 mt-1.5">
          <span style={{ fontSize: '10px', color: trendUp ? 'var(--green)' : trendDown ? 'var(--red)' : 'var(--text-muted)' }}>
            {trendUp ? '▲' : trendDown ? '▼' : '—'}
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
            {Math.abs(trend).toFixed(1)} vs prev
          </span>
        </div>
      )}
    </div>
  );
}