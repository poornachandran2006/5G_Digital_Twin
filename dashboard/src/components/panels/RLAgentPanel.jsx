import { useSim } from '../../context/SimContext';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ACTION_NAMES, CELL_COLORS } from '../../constants';
import PanelWrapper from '../PanelWrapper';

const BAR_COLORS = ['var(--text-muted)', 'var(--accent)', 'var(--purple)', 'var(--green)'];

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
    name:  ACTION_NAMES[Number(k)],
    count: v,
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
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>4 actions:</span> NoOp · LoadBalance (soft handover) · PowerCtrl (adjust TX power) · Handover (hard UE reassignment for critical overload).
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            Stable-Baselines3 · MlpPolicy · 2×64 hidden · 200K steps · lr=3e-4
          </p>
        </>
      }
    >
      <div className="space-y-4">
        {/* Current actions table */}
        <div className="chart-card">
          <p className="text-xs font-mono mb-3" style={{ color: 'var(--text-muted)' }}>CURRENT ACTIONS</p>
          <table className="w-full text-sm">
            <thead>
              <tr
                className="text-xs font-mono"
                style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}
              >
                <th className="text-left pb-2">Cell</th>
                <th className="text-left pb-2">Action ID</th>
                <th className="text-left pb-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2].map((i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="py-2 font-mono" style={{ color: CELL_COLORS[i] }}>Cell {i}</td>
                  <td className="py-2 font-mono" style={{ color: 'var(--text-secondary)' }}>{actions[i] ?? '—'}</td>
                  <td className="py-2 font-mono" style={{ color: 'var(--text-secondary)' }}>{ACTION_NAMES[actions[i]] ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Action distribution chart */}
        <div className="chart-card">
          <p className="text-xs font-mono mb-3" style={{ color: 'var(--text-muted)' }}>
            ACTION DISTRIBUTION — LAST {state.ticks.length} TICKS
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'var(--text-secondary)' }}
              />
              <Bar dataKey="count" isAnimationActive={false} radius={[4, 4, 0, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={BAR_COLORS[d.id]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </PanelWrapper>
  );
}