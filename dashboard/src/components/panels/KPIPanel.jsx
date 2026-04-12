import { useState } from 'react';
import { useSim } from '../../context/SimContext';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CELL_COLORS } from '../../constants';

const METRICS = [
  { key: 'throughput_mbps', label: 'Throughput', unit: 'Mbps' },
  { key: 'latency_ms', label: 'Latency', unit: 'ms' },
  { key: 'load_percent', label: 'Cell Load', unit: '%' },
];

export default function KPIPanel() {
  const { state } = useSim();
  const [metric, setMetric] = useState('throughput_mbps');
  const ticks = state.ticks.slice(-60);
  const chartData = ticks.map((t) => ({
    tick: t.tick,
    cell0: t.cells?.[0]?.[metric] ?? 0,
    cell1: t.cells?.[1]?.[metric] ?? 0,
    cell2: t.cells?.[2]?.[metric] ?? 0,
  }));
  const cur = METRICS.find((m) => m.key === metric);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">KPI Time-Series</h2>
        <div className="flex gap-2 ml-4">
          {METRICS.map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMetric(m.key)}
              className={`px-3 py-1 text-xs rounded border transition-colors font-mono
                ${metric === m.key
                  ? 'bg-[#00d4ff]/20 text-[#00d4ff] border-[#00d4ff]/40'
                  : 'text-gray-400 border-gray-700 hover:border-gray-500'}`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
      <div className="bg-[#111827] rounded-xl border border-gray-800 p-4">
        <p className="text-xs text-gray-500 font-mono mb-3">
          {cur?.label} ({cur?.unit}) — Last 60 ticks
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <XAxis dataKey="tick" tick={{ fill: '#6b7280', fontSize: 10 }} />
            <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            {[0, 1, 2].map((i) => (
              <Line
                key={i}
                type="monotone"
                dataKey={`cell${i}`}
                name={`Cell ${i}`}
                stroke={CELL_COLORS[i]}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
