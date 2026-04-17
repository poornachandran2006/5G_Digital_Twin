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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

        {/* Stat cards — 2×2 on mobile, 4×1 on desktop */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '10px',
        }}
          className="overview-stat-grid"
        >
          <StatCard
            label="Total Throughput"
            value={tp}
            unit="Mbps"
            trend={tp - (prev?.kpis?.total_throughput ?? tp)}
            highlight
          />
          <StatCard
            label="Mean Latency"
            value={lat}
            unit="ms"
            trend={-(lat - (prev?.kpis?.mean_latency ?? lat))}
            thresholdWarn={50}
            thresholdCrit={80}
          />
          <StatCard
            label="Handovers / tick"
            value={ho}
            unit=""
            thresholdWarn={3}
            thresholdCrit={6}
          />
          <StatCard
            label="Active UEs"
            value={ues}
            unit="/ 20"
            highlight
          />
        </div>

        {/* Per-cell load cards — stack on mobile, 3-col on md+ */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(1, 1fr)',
          gap: '10px',
        }}
          className="overview-cell-grid"
        >
          {[0, 1, 2].map((i) => {
            const cell = cur?.cells?.[i];
            const load = cell?.load_percent ?? 0;
            return (
              <div key={i} className="chart-card" style={{ padding: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', fontFamily: 'monospace', color: CELL_COLORS[i], fontWeight: 700 }}>
                    gNB Cell {i}
                  </span>
                  <span
                    style={{
                      fontSize: '11px',
                      padding: '2px 8px',
                      borderRadius: '999px',
                      fontFamily: 'monospace',
                      fontWeight: 700,
                      background: `${STATUS_COLOR(load)}22`,
                      color: STATUS_COLOR(load),
                    }}
                  >
                    {(load * 100).toFixed(0)}%
                  </span>
                </div>
                {/* Load bar */}
                <div style={{ width: '100%', height: '4px', background: 'var(--border)', borderRadius: '2px', marginBottom: '8px' }}>
                  <div style={{
                    height: '4px',
                    borderRadius: '2px',
                    width: `${load * 100}%`,
                    background: STATUS_COLOR(load),
                    transition: 'width 0.4s ease',
                  }} />
                </div>
                <ResponsiveContainer width="100%" height={45}>
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
        <div className="chart-card" style={{ padding: '14px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', marginBottom: '10px', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            TOTAL THROUGHPUT — LAST 60 TICKS
          </p>
          <ResponsiveContainer width="100%" height={110}>
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
                  fontSize: '12px',
                }}
                labelStyle={{ color: 'var(--text-secondary)' }}
                itemStyle={{ color: 'var(--accent)' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <style>{`
        @media (min-width: 640px) {
          .overview-cell-grid {
            grid-template-columns: repeat(3, 1fr) !important;
          }
        }
        @media (min-width: 768px) {
          .overview-stat-grid {
            grid-template-columns: repeat(4, 1fr) !important;
          }
        }
      `}</style>
    </PanelWrapper>
  );
}