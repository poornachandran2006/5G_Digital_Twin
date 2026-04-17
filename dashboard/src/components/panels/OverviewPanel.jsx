import { useSim } from '../../context/SimContext';
import StatCard from '../shared/StatCard';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { CELL_COLORS, STATUS_COLOR } from '../../constants';
import PanelWrapper from '../PanelWrapper';

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
    <PanelWrapper
      title="Network Overview"
      icon="◉"
      description="Real-time system-wide KPIs across all 3 gNB base stations and 20 mobile UEs. Throughput, latency, and cell load update every simulation tick (1 second)."
      hint="Click ⓘ to understand each metric"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            The Overview panel is what a <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Network Operations Centre (NOC)</span> engineer sees first — the health of the entire 5G network at a glance.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Total Throughput</span> — sum of all data delivered across all cells. Drops when congestion reduces scheduler efficiency.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Mean Latency</span> — modelled as 15 + 85 × cell_load ms. At 100% load, latency hits 100ms — the edge of acceptable for real-time applications.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Cell Load bars</span> — percentage of Physical Resource Blocks (PRBs) in use. Each gNB has 100 PRBs. Warning at 70%, critical at 90%.
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            Simulation: SimPy · 1s ticks · 3-hour run · 3.5 GHz n78 band
          </p>
        </>
      }
    >
      <div className="space-y-4">
        {/* Stat cards */}
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

        {/* Per-cell load cards */}
        <div className="grid grid-cols-3 gap-3">
          {[0, 1, 2].map((i) => {
            const cell = cur?.cells?.[i];
            const load = cell?.load_percent ?? 0;
            return (
              <div key={i} className="chart-card">
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

        {/* Total throughput chart */}
        <div className="chart-card">
          <p className="text-xs font-mono mb-3" style={{ color: 'var(--text-muted)' }}>
            TOTAL THROUGHPUT — LAST 60 TICKS
          </p>
          <ResponsiveContainer width="100%" height={120}>
            <AreaChart data={ticks.map((t) => ({ v: t.kpis?.total_throughput ?? 0, tick: t.tick }))}>
              <Area
                type="monotone"
                dataKey="v"
                stroke="var(--accent)"
                fill="var(--accent-dim)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'var(--text-secondary)' }}
                itemStyle={{ color: 'var(--accent)' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </PanelWrapper>
  );
}