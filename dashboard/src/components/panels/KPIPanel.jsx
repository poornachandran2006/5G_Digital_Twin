import { useState } from 'react';
import InfoModal from '../InfoModal';
import { useSim } from '../../context/SimContext';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CELL_COLORS } from '../../constants';
import PanelWrapper from '../PanelWrapper';

const METRICS = [
  { key: 'throughput_mbps', label: 'Throughput', unit: 'Mbps' },
  { key: 'latency_ms', label: 'Latency', unit: 'ms' },
  { key: 'load_percent', label: 'Cell Load', unit: '%' },
];

export default function KPIPanel() {
  const { state } = useSim();
  const [metric, setMetric] = useState('throughput_mbps');
  const [showInfo, setShowInfo] = useState(false);
  const ticks = state.ticks.slice(-60);
  const chartData = ticks.map((t) => ({
    tick: t.tick,
    cell0: t.cells?.[0]?.[metric] ?? 0,
    cell1: t.cells?.[1]?.[metric] ?? 0,
    cell2: t.cells?.[2]?.[metric] ?? 0,
  }));
  const cur = METRICS.find((m) => m.key === metric);




  return (
    <PanelWrapper
      title="KPI Time-Series"
      icon="△"
      description="Per-cell throughput, latency, and PRB load plotted over the last 60 ticks. Use the buttons to switch metrics. Each coloured line is one gNB tower."
      hint="Switch metrics with the buttons below"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            KPIs are computed per tick by <span style={{ color: 'var(--accent)', fontWeight: 600 }}>kpi/calculator.py</span> from the raw SimPy simulation state.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Throughput</span> — sum of Shannon capacity across all UEs on a cell: B × log₂(1 + SINR), where B scales with allocated PRBs.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Latency</span> — modelled as 15 + 85 × load ms. Below 50ms is healthy. Above 80ms the cell is struggling under load.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Cell Load</span> — PRBs used ÷ 100 (max). Warning threshold: 70%. Critical threshold: 90%. The LSTM watches this signal as its primary input.
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            Last 60 ticks shown · 1 tick = 1 second · Recharts
          </p>
        </>
      }
    >
      <div className="space-y-4">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">KPI Time-Series</h2>
              <button
                onClick={() => setShowInfo(true)}
                className="text-gray-500 hover:text-green-400 transition-colors text-sm"
                title="What is this panel?"
              >
                ⓘ
              </button>
            </div>

            {showInfo && (
              <InfoModal title="KPI Time-Series Panel" onClose={() => setShowInfo(false)}>
                <p>
                  This panel shows the three most important <span className="text-green-400 font-semibold">Key Performance Indicators</span> for each cell tower, plotted live over the last 60 seconds.
                </p>
                <p>
                  <span className="text-white font-semibold">Throughput (Mbps)</span> — total data delivered by each cell per second. Drops sharply when a cell becomes congested.
                </p>
                <p>
                  <span className="text-white font-semibold">Latency (ms)</span> — average delay experienced by users on that cell. Below 50ms is healthy; above 80ms means the cell is struggling.
                </p>
                <p>
                  <span className="text-white font-semibold">Cell Load (%)</span> — percentage of Physical Resource Blocks (PRBs) in use. Each cell has 100 PRBs max. Warning at 70%, critical at 90%.
                </p>
                <p className="text-gray-500 text-xs pt-1">
                  Each colored line is one cell tower. Use the buttons to switch between metrics. Data updates every tick (1 second).
                </p>
              </InfoModal>
            )}
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
      </div>
    </PanelWrapper>
  );
}