const NAV = [
  { id: 'overview', label: 'Overview', icon: '◉' },
  { id: 'map', label: 'Network Map', icon: '⬡' },
  { id: 'kpi', label: 'KPIs', icon: '△' },
  { id: 'predictions', label: 'Predictions', icon: '◈' },
  { id: 'shap', label: 'SHAP', icon: '⬢' },
  { id: 'anomaly', label: 'Anomaly', icon: '⚠' },
  { id: 'abtest', label: 'A/B Test', icon: '⚖' },
  { id: 'rl', label: 'RL Agent', icon: '⬟' },
];

export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="w-48 bg-[#0d1424] border-r border-gray-800 flex flex-col py-6 gap-1 shrink-0">
      <div className="px-4 mb-6">
        <p className="text-[10px] text-gray-500 uppercase tracking-widest">5G Digital Twin</p>
      </div>
      {NAV.map((n) => (
        <button
          key={n.id}
          type="button"
          onClick={() => onNavigate(n.id)}
          className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors text-left
            ${active === n.id
              ? 'bg-[#00d4ff]/10 text-[#00d4ff] border-r-2 border-[#00d4ff]'
              : 'text-gray-400 hover:text-white hover:bg-white/5'}`}
        >
          <span>{n.icon}</span>
          {n.label}
        </button>
      ))}
    </aside>
  );
}
