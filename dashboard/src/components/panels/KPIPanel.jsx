import { useSim } from '../../context/SimContext';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CELL_COLORS } from '../../constants';
import PanelWrapper from '../PanelWrapper';
import { useState } from 'react';

const METRICS = [
  { key: 'throughput_mbps', label: 'Throughput', unit: 'Mbps' },
  { key: 'latency_ms',      label: 'Latency',    unit: 'ms'   },
  { key: 'load_percent',    label: 'Cell Load',  unit: '%'    },
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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

        {/* Metric selector — full-width row, wraps on small screens */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {METRICS.map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMetric(m.key)}
              style={{
                padding: '6px 14px',
                fontSize: '12px',
                borderRadius: '6px',
                fontFamily: 'monospace',
                cursor: 'pointer',
                transition: 'all 0.15s',
                flex: '1 1 auto',
                minWidth: '90px',
                background: metric === m.key ? 'var(--accent-dim)' : 'transparent',
                border: `1px solid ${metric === m.key ? 'var(--accent)' : 'var(--border-accent)'}`,
                color: metric === m.key ? 'var(--accent)' : 'var(--text-muted)',
              }}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Chart card */}
        <div className="chart-card" style={{ padding: '14px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', marginBottom: '10px', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            {cur?.label} ({cur?.unit}) — Last 60 ticks
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
              <XAxis
                dataKey="tick"
                tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: 'var(--border)' }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={36}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                labelStyle={{ color: 'var(--text-secondary)' }}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', color: 'var(--text-secondary)', paddingTop: '8px' }}
              />
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
    </PanelWrapper>
  );
}