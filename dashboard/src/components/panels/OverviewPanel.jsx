import { useState } from "react";
import InfoModal from "../InfoModal";
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
  const [showInfo, setShowInfo] = useState(false);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-bold text-white">Overview</h2>
        <button
          onClick={() => setShowInfo(true)}
          className="text-gray-500 hover:text-green-400 transition-colors text-sm"
          title="What is this panel?"
        >
          ⓘ
        </button>
      </div>

      {showInfo && (
        <InfoModal title="Overview Panel" onClose={() => setShowInfo(false)}>
          <p>
            This panel shows the <span className="text-green-400 font-semibold">system-wide health</span> of
            the entire 5G network at a glance — updated every second.
          </p>
          <p>
            <span className="text-white font-semibold">Total Throughput</span> — the combined data speed
            (in Mbps) being delivered to all 20 users across all 3 towers right now.
          </p>
          <p>
            <span className="text-white font-semibold">Mean Latency</span> — the average delay (in
            milliseconds) a user experiences. Below 50ms is good; above 80ms signals congestion.
          </p>
          <p>
            <span className="text-white font-semibold">Cell Load Bars</span> — each bar shows how busy
            that tower is (0–100% of its radio resources used). Above 70% = warning. Above 90% = critical.
          </p>
          <p className="text-gray-500 text-xs pt-1">
            Formula: Cell Load = PRBs allocated ÷ max PRBs (100). PRB = Physical Resource Block,
            the basic unit of radio capacity in 5G NR.
          </p>
        </InfoModal>
      )}
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
