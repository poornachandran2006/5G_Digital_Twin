import { useSim } from '../../context/SimContext';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  BarChart, Bar, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { useState } from 'react';
import InfoModal from '../InfoModal';
import PanelWrapper from '../PanelWrapper';

const ACTION_LABELS = ['No-Op', 'LoadBalance', 'MassBalance', 'Emergency'];
const ACTION_COLORS = ['#64748b', '#22c55e', '#f59e0b', '#ef4444'];

export default function ABTestPanel() {
  const [showInfo, setShowInfo] = useState(false);
  const { state } = useSim();
  const { ticks } = state;

  // Build chart data from last 60 ticks
  const chartData = ticks.slice(-60).map((t) => ({
    tick: t.tick,
    ppo: t.ab_comparison?.ppo_reward ?? 0,
    rb: t.ab_comparison?.rb_reward ?? 0,
  }));

  // Compute running summary from all ticks in buffer
  const validTicks = ticks.filter((t) => t.ab_comparison?.tick !== undefined);
  const totalTicks = validTicks.length;

  const ppoRewards = validTicks.map((t) => t.ab_comparison.ppo_reward);
  const rbRewards = validTicks.map((t) => t.ab_comparison.rb_reward);

  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const ppoAvg = avg(ppoRewards);
  const rbAvg = avg(rbRewards);
  const ppoWins = ppoRewards.filter((p, i) => p > rbRewards[i]).length;
  const winRate = totalTicks ? ((ppoWins / totalTicks) * 100).toFixed(1) : '0.0';

  // Action distribution from last 60 ticks
  const actionCounts = { ppo: [0, 0, 0, 0], rb: [0, 0, 0, 0] };
  ticks.slice(-60).forEach((t) => {
    const ab = t.ab_comparison;
    if (!ab) return;
    if (ab.ppo_action >= 0 && ab.ppo_action <= 3) actionCounts.ppo[ab.ppo_action]++;
    if (ab.rb_action >= 0 && ab.rb_action <= 3) actionCounts.rb[ab.rb_action]++;
  });

  const actionData = ACTION_LABELS.map((label, i) => ({
    action: label,
    PPO: actionCounts.ppo[i],
    Rule: actionCounts.rb[i],
  }));

  // Latest tick actions
  const latest = ticks[ticks.length - 1]?.ab_comparison;
  const ppoActionNow = latest ? ACTION_LABELS[latest.ppo_action] ?? 'No-Op' : '—';
  const rbActionNow = latest ? ACTION_LABELS[latest.rb_action] ?? 'No-Op' : '—';

  return (
    <PanelWrapper
      title="A/B Testing — PPO vs Rule-Based"
      icon="⚖"
      description="Live experiment: both agents observe the same network state every tick and take independent actions. Their rewards are compared to quantify how much better the RL agent is than a hand-coded policy."
      hint="Higher reward = better load balancing decision"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            Both agents run simultaneously in <span style={{ color: 'var(--accent)', fontWeight: 600 }}>shadow mode</span> — the rule-based agent's decisions are simulated but never applied to the actual network. Only the PPO agent's actions take effect.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Rule-Based policy:</span> fixed thresholds — if any cell exceeds 90% trigger emergency rebalancing, above 80% trigger mass balance, spread greater than 30% trigger load balance. Simple and predictable, but rigid and prone to overreaction.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>PPO Agent:</span> trained for 200,000 steps via Stable-Baselines3. It learned when NOT to intervene — choosing NoOp on healthy ticks while the rule-based agent triggers unnecessary rebalancing, which itself introduces handover overhead.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Reward:</span> +0.2 per cell below 70% load, −0.5 at warning (70–90%), −1.0 at critical (90%+), plus bonus for balanced load distribution across cells.
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            PPO · MlpPolicy 2×64 · Gymnasium · Discrete(4) action space
          </p>
        </>
      }
    >
      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-bold text-white">A/B Testing — PPO vs Rule-Based</h2>
          <button
            onClick={() => setShowInfo(true)}
            className="text-gray-500 hover:text-green-400 transition-colors text-sm"
            title="What is this panel?"
          >
            ⓘ
          </button>
        </div>

        {showInfo && (
          <InfoModal title="A/B Testing Panel" onClose={() => setShowInfo(false)}>
            <p>
              This panel runs a <span className="text-green-400 font-semibold">live experiment</span>: the PPO reinforcement learning agent and a hand-coded rule-based policy are both applied to the same network state every tick, and their decisions are compared.
            </p>
            <p>
              <span className="text-white font-semibold">PPO Agent</span> was trained with Stable-Baselines3 for 200,000 steps. It learned load balancing purely from trial and error — it was never told any rules.
            </p>
            <p>
              <span className="text-white font-semibold">Rule-Based policy</span> uses fixed thresholds: if any cell exceeds 90% load, trigger emergency rebalancing. Simple and predictable, but rigid.
            </p>
            <p>
              <span className="text-white font-semibold">Reward formula:</span> throughput bonus − latency penalty − handover cost. Higher is better. PPO's win rate shows how often it outperforms the rule-based approach per tick.
            </p>
            <p className="text-gray-500 text-xs pt-1">
              Both agents run in shadow mode — only the PPO agent's decisions are actually applied to the network. The rule-based agent is simulated in parallel for comparison only.
            </p>
          </InfoModal>
        )}
        <p className="text-xs text-slate-400">
          PPO (Stable-Baselines3, 200k steps) vs threshold policy running in parallel.
          Same environment, same reward formula. Divergence appears during congestion events.
        </p>

        {/* Summary cards */}
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">PPO Avg Reward</p>
            <p className="text-xl font-bold text-indigo-400">{ppoAvg.toFixed(3)}</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">Rule Avg Reward</p>
            <p className="text-xl font-bold text-amber-400">{rbAvg.toFixed(3)}</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">PPO Win Rate</p>
            <p className="text-xl font-bold text-green-400">{winRate}%</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">Ticks Compared</p>
            <p className="text-xl font-bold text-white">{totalTicks}</p>
          </div>
        </div>

        {/* Current action row */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-indigo-900/30 border border-indigo-500/40 rounded-lg p-3">
            <p className="text-xs text-slate-400 mb-1">PPO Action Now</p>
            <p className="text-lg font-bold text-indigo-300">{ppoActionNow}</p>
          </div>
          <div className="bg-amber-900/30 border border-amber-500/40 rounded-lg p-3">
            <p className="text-xs text-slate-400 mb-1">Rule-Based Action Now</p>
            <p className="text-lg font-bold text-amber-300">{rbActionNow}</p>
          </div>
        </div>

        {/* Reward time-series */}
        <div>
          <p className="text-xs text-slate-400 mb-2">Reward per Tick — Last 60 Ticks</p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="tick" tick={{ fontSize: 10, fill: '#94a3b8' }} interval={9} />
              <YAxis domain={[-1, 1]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', fontSize: 12 }}
                formatter={(v, name) => [v.toFixed(3), name === 'ppo' ? 'PPO' : 'Rule-Based']}
              />
              <Legend
                formatter={(val) => val === 'ppo' ? 'PPO Agent' : 'Rule-Based'}
                wrapperStyle={{ fontSize: 11 }}
              />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 2" />
              <Line type="monotone" dataKey="ppo" stroke="#818cf8" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="rb" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Action distribution */}
        <div>
          <p className="text-xs text-slate-400 mb-2">Action Distribution — Last 60 Ticks</p>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={actionData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="action" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="PPO" fill="#818cf8" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Rule" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-500 mt-1">
            Both agents choose No-Op when network is healthy. Divergence appears during congestion.
          </p>
        </div>
      </div>
    </PanelWrapper>
  );
}