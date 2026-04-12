import { useSim } from '../../context/SimContext';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ACTION_NAMES, CELL_COLORS } from '../../constants';

export default function RLAgentPanel() {
  const { state } = useSim();
  const actions = state.currentTick?.ppo_actions ?? {};
  const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
  state.ticks.forEach((t) => {
    Object.values(t.ppo_actions ?? {}).forEach((a) => {
      counts[a] = (counts[a] || 0) + 1;
    });
  });
  const chartData = Object.entries(counts).map(([k, v]) => ({
    name: ACTION_NAMES[Number(k)],
    count: v,
    id: Number(k),
  }));
  const barColors = ['#6b7280', '#00d4ff', '#8b5cf6', '#10b981'];
  return (
    <div className="space-y-4">
      <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">PPO RL Agent</h2>
      <div className="bg-[#111827] rounded-xl border border-gray-800 p-4">
        <p className="text-xs text-gray-500 font-mono mb-3">CURRENT ACTIONS</p>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 font-mono border-b border-gray-700">
              <th className="text-left pb-2">Cell</th>
              <th className="text-left pb-2">Action ID</th>
              <th className="text-left pb-2">Action</th>
            </tr>
          </thead>
          <tbody>
            {[0, 1, 2].map((i) => (
              <tr key={i} className="border-b border-gray-800">
                <td className="py-2 font-mono" style={{ color: CELL_COLORS[i] }}>Cell {i}</td>
                <td className="py-2 font-mono text-gray-300">{actions[i] ?? '—'}</td>
                <td className="py-2 font-mono text-gray-300">{ACTION_NAMES[actions[i]] ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="bg-[#111827] rounded-xl border border-gray-800 p-4">
        <p className="text-xs text-gray-500 font-mono mb-3">
          ACTION DISTRIBUTION — LAST {state.ticks.length} TICKS
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData}>
            <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} />
            <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <Bar dataKey="count" isAnimationActive={false} radius={[4, 4, 0, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={barColors[d.id]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
