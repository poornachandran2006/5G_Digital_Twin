export default function StatCard({ label, value, unit, trend }) {
  return (
    <div className="bg-[#111827] rounded-xl p-4 border border-gray-800">
      <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">{label}</p>
      <div className="flex items-end gap-1">
        <span className="text-2xl font-mono text-white">{value ?? '—'}</span>
        {unit && <span className="text-sm text-gray-500 mb-0.5">{unit}</span>}
        {trend !== undefined && (
          <span className={`text-xs mb-1 ml-1 ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend >= 0 ? '▲' : '▼'}
          </span>
        )}
      </div>
    </div>
  );
}
