export default function StatCard({ label, value, unit, trend }) {
  return (
    <div
      className="rounded-xl p-4 border"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
      }}
    >
      <p
        className="text-xs uppercase tracking-wider mb-1"
        style={{ color: 'var(--text-secondary)' }}
      >
        {label}
      </p>

      <div className="flex items-end gap-1">
        <span
          className="text-2xl font-mono"
          style={{ color: 'var(--text-primary)' }}
        >
          {value ?? '—'}
        </span>

        {unit && (
          <span
            className="text-sm mb-0.5"
            style={{ color: 'var(--text-muted)' }}
          >
            {unit}
          </span>
        )}

        {trend !== undefined && (
          <span
            className={`text-xs mb-1 ml-1 ${
              trend >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {trend >= 0 ? '▲' : '▼'}
          </span>
        )}
      </div>
    </div>
  );
}