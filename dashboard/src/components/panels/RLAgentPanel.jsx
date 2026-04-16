import { useSim } from '../../context/SimContext';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ACTION_NAMES, CELL_COLORS } from '../../constants';
import { useState } from 'react';
import InfoModal from '../InfoModal';
import PanelWrapper from '../PanelWrapper';

export default function RLAgentPanel() {
  const [showInfo, setShowInfo] = useState(false);
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
    <PanelWrapper
      title="PPO RL Agent"
      icon="⬟"
      description="Live decisions of the Proximal Policy Optimization agent. It chooses one of 4 actions per cell every tick based on load, congestion probability, and UE distribution — entirely learned from 200,000 steps of trial and error."
      hint="A healthy network should show mostly No-Op"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>PPO (Proximal Policy Optimization)</span> is a state-of-the-art RL algorithm used in real telecom research and by OpenAI for training GPT policies. It learns by collecting environment interactions, computing a clipped policy gradient update, and iterating — never diverging too far from the current policy in one step.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Observation space (9-dim):</span> cell0/1/2 load, congestion probability per cell, UE ratio per cell. All normalized to [0, 1].
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>4 actions:</span> NoOp · LoadBalance (soft handover, redirect UEs from busiest to least loaded) · PowerCtrl (adjust TX power) · Handover (hard UE reassignment for critical overload).
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            The action distribution chart shows the agent's policy in practice. Mostly NoOp signals a well-trained agent that intervenes precisely, not constantly.
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            Stable-Baselines3 · MlpPolicy · 2×64 hidden · 200K steps · lr=3e-4
          </p>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">PPO RL Agent</h2>
          <button
            onClick={() => setShowInfo(true)}
            className="text-gray-500 hover:text-green-400 transition-colors text-sm"
            title="What is this panel?"
          >
            ⓘ
          </button>
        </div>

        {showInfo && (
          <InfoModal title="PPO RL Agent Panel" onClose={() => setShowInfo(false)}>
            <p>
              This panel shows the live decisions of the <span className="text-green-400 font-semibold">PPO reinforcement learning agent</span> — the brain of the network optimizer. PPO stands for Proximal Policy Optimization, a state-of-the-art RL algorithm used in real telecom research.
            </p>
            <p>
              <span className="text-white font-semibold">How it was trained:</span> The agent played 200,000 simulated seconds of network management, receiving rewards for high throughput and low latency, and penalties for congestion and unnecessary handovers. It learned entirely through trial and error.
            </p>
            <p>
              <span className="text-white font-semibold">4 possible actions per cell:</span> NoOp (do nothing), LoadBalance (redirect users to less busy cell), PowerCtrl (adjust transmit power), Handover (force a UE to switch towers).
            </p>
            <p>
              <span className="text-white font-semibold">Action distribution</span> shows which actions the agent has favoured across the entire session. A healthy network should mostly show NoOp — intervention only when needed.
            </p>
            <p className="text-gray-500 text-xs pt-1">
              Framework: Stable-Baselines3. Environment: custom Gymnasium env wrapping the SimPy simulation. Reward function: throughput bonus − latency penalty − 0.1 × handover cost.
            </p>
          </InfoModal>
        )}
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
    </PanelWrapper>
  );
}