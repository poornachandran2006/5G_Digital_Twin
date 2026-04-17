import { useSim } from '../../context/SimContext';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ACTION_NAMES, CELL_COLORS } from '../../constants';
import PanelWrapper from '../PanelWrapper';

const BAR_COLORS = ['var(--text-muted)', 'var(--accent)', 'var(--purple)', 'var(--green)'];
const ACTION_DESCS = [
  'No intervention — network is healthy',
  'Soft handover: move UEs from busiest to least-loaded cell',
  'Aggressive rebalancing across all 3 cells simultaneously',
  'Hard handover for critical overload (≥90% PRB load)',
];

export default function RLAgentPanel() {
  const { state } = useSim();
  const actions = state.currentTick?.ppo_actions ?? {};

  const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
  state.ticks.forEach((t) => {
    Object.values(t.ppo_actions ?? {}).forEach((a) => {
      counts[a] = (counts[a] || 0) + 1;
    });
  });

  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;

  const chartData = Object.entries(counts).map(([k, v]) => ({
    name:  ACTION_NAMES[Number(k)],
    count: v,
    pct:   ((v / total) * 100).toFixed(1),
    id:    Number(k),
  }));

  return (
    <PanelWrapper
      title="PPO RL Agent"
      icon="⬟"
      description="Live decisions of the Proximal Policy Optimization agent. It chooses one of 4 actions per cell every tick based on load, congestion probability, and UE distribution — entirely learned from 200,000 steps of trial and error."
      hint="A healthy network should show mostly No-Op"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>PPO (Proximal Policy Optimization)</span> is a state-of-the-art RL algorithm used in real telecom research and by OpenAI for training GPT policies. It learns by collecting environment interactions, computing a clipped policy gradient update, and iterating.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Observation space (9-dim):</span> cell load × 3, congestion probability × 3, UE ratio × 3. All normalized to [0, 1].
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>4 actions:</span> NoOp · LoadBalance (soft handover) · MassBalance (all cells) · EmergencyHandover (critical overload).
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            Stable-Baselines3 · MlpPolicy · 2×64 hidden · 200K steps · lr=3e-4
          </p>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

        {/* Current actions table — full-width, scrolls if needed */}
        <div className="chart-card" style={{ padding: '14px', overflowX: 'auto' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            CURRENT PPO ACTIONS
          </p>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', minWidth: '280px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Cell', 'Action ID', 'Action', 'Description'].map((h) => (
                  <th key={h} style={{
                    textAlign: 'left',
                    paddingBottom: '8px',
                    fontSize: '10px',
                    fontFamily: 'monospace',
                    color: 'var(--text-muted)',
                    fontWeight: 600,
                    whiteSpace: 'nowrap',
                    paddingRight: '12px',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2].map((i) => {
                const actionId = actions[i] ?? 0;
                return (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '8px 12px 8px 0', fontFamily: 'monospace', color: CELL_COLORS[i], fontWeight: 700, whiteSpace: 'nowrap' }}>
                      Cell {i}
                    </td>
                    <td style={{ padding: '8px 12px 8px 0', fontFamily: 'monospace', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                      {actionId}
                    </td>
                    <td style={{ padding: '8px 12px 8px 0', fontFamily: 'monospace', color: BAR_COLORS[actionId], fontWeight: 600, whiteSpace: 'nowrap' }}>
                      {ACTION_NAMES[actionId] ?? '—'}
                    </td>
                    <td style={{ padding: '8px 0', fontSize: '11px', color: 'var(--text-muted)' }}>
                      {ACTION_DESCS[actionId] ?? '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Action distribution chart */}
        <div className="chart-card" style={{ padding: '14px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            ACTION DISTRIBUTION — ALL {state.ticks.length} TICKS
          </p>
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis
                dataKey="name"
                tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '8px',
                  fontSize: 12,
                }}
                labelStyle={{ color: 'var(--text-secondary)' }}
                formatter={(v, _, props) => [`${v} ticks (${props.payload.pct}%)`, 'Count']}
              />
              <Bar dataKey="count" isAnimationActive={false} radius={[4, 4, 0, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={BAR_COLORS[d.id]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Legend row */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px', borderTop: '1px solid var(--border)', paddingTop: '10px' }}>
            {chartData.map((d) => (
              <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px' }}>
                <span style={{ width: 10, height: 10, borderRadius: '2px', background: BAR_COLORS[d.id], display: 'inline-block', flexShrink: 0 }} />
                <span style={{ color: 'var(--text-secondary)' }}>{d.name}</span>
                <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>({d.pct}%)</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </PanelWrapper>
  );
}