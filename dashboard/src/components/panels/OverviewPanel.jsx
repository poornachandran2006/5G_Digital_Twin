import { useSim } from '../../context/SimContext';
import StatCard from '../shared/StatCard';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { CELL_COLORS, STATUS_COLOR } from '../../constants';

export default function OverviewPanel() {
  const { state } = useSim();
  const cur = state.currentTick;
  const ticks = state.ticks.slice(-60);
  const prev = state.ticks[state.ticks.length - 2];
  const tp = cur?.kpis?.total_throughput ?? 0;
  const lat = cur?.kpis?.mean_latency ?? 0;
  const ho = cur?.kpis?.handover_count ?? 0;
  const ues = cur?.kpis?.active_ues ?? 0;
  return (
    <div className="space-y-4">
      <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">Overview</h2>
      <div className="grid grid-cols-4 gap-3">
        <StatCard
          label="Total Throughput"
          value={tp}
          unit="Mbps"
          trend={tp - (prev?.kpis?.total_throughput ?? tp)}
        />
        <StatCard
          label="Mean Latency"
          value={lat}
          unit="ms"
          trend={-(lat - (prev?.kpis?.mean_latency ?? lat))}
        />
        <StatCard label="Handovers/tick" value={ho} unit="" />
        <StatCard label="Active UEs" value={ues} unit="" />
      </div>
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => {
          const cell = cur?.cells?.[i];
          const load = cell?.load_percent ?? 0;
          return (
            <div key={i} className="bg-[#111827] rounded-xl p-4 border border-gray-800">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-mono" style={{ color: CELL_COLORS[i] }}>Cell {i}</span>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-mono"
                  style={{ background: `${STATUS_COLOR(load)}22`, color: STATUS_COLOR(load) }}
                >
                  {(load * 100).toFixed(0)}%
                </span>
              </div>
              <ResponsiveContainer width="100%" height={50}>
                <AreaChart data={ticks.map((t) => ({ v: t.cells?.[i]?.load_percent ?? 0 }))}>
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke={CELL_COLORS[i]}
                    fill={`${CELL_COLORS[i]}22`}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
      <div className="bg-[#111827] rounded-xl p-4 border border-gray-800">
        <p className="text-xs text-gray-400 mb-2 font-mono">TOTAL THROUGHPUT — LAST 60 TICKS</p>
        <ResponsiveContainer width="100%" height={120}>
          <AreaChart data={ticks.map((t) => ({ v: t.kpis?.total_throughput ?? 0, tick: t.tick }))}>
            <Area
              type="monotone"
              dataKey="v"
              stroke="#00d4ff"
              fill="#00d4ff22"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#9ca3af' }}
              itemStyle={{ color: '#00d4ff' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
