import { useSim } from '../../context/SimContext';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  BarChart, Bar, ResponsiveContainer, ReferenceLine
} from 'recharts';
import PanelWrapper from '../PanelWrapper';

const ACTION_LABELS = ['No-Op', 'LoadBalance', 'MassBalance', 'Emergency'];

export default function ABTestPanel() {
  const { state } = useSim();
  const { ticks } = state;

  const chartData = ticks.slice(-60).map((t) => ({
    tick: t.tick,
    ppo: t.ab_comparison?.ppo_reward ?? 0,
    rb:  t.ab_comparison?.rb_reward  ?? 0,
  }));

  const validTicks  = ticks.filter((t) => t.ab_comparison?.tick !== undefined);
  const totalTicks  = validTicks.length;
  const ppoRewards  = validTicks.map((t) => t.ab_comparison.ppo_reward);
  const rbRewards   = validTicks.map((t) => t.ab_comparison.rb_reward);
  const avg         = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const ppoAvg      = avg(ppoRewards);
  const rbAvg       = avg(rbRewards);
  const ppoWins     = ppoRewards.filter((p, i) => p > rbRewards[i]).length;
  const winRate     = totalTicks ? ((ppoWins / totalTicks) * 100).toFixed(1) : '0.0';

  const actionCounts = { ppo: [0, 0, 0, 0], rb: [0, 0, 0, 0] };
  ticks.slice(-60).forEach((t) => {
    const ab = t.ab_comparison;
    if (!ab) return;
    if (ab.ppo_action >= 0 && ab.ppo_action <= 3) actionCounts.ppo[ab.ppo_action]++;
    if (ab.rb_action  >= 0 && ab.rb_action  <= 3) actionCounts.rb[ab.rb_action]++;
  });

  const actionData = ACTION_LABELS.map((label, i) => ({
    action: label,
    PPO:  actionCounts.ppo[i],
    Rule: actionCounts.rb[i],
  }));

  const latest      = ticks[ticks.length - 1]?.ab_comparison;
  const ppoActionNow = latest ? ACTION_LABELS[latest.ppo_action] ?? 'No-Op' : '—';
  const rbActionNow  = latest ? ACTION_LABELS[latest.rb_action]  ?? 'No-Op' : '—';

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
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>PPO Agent:</span> trained for 200,000 steps. It learned when NOT to intervene — choosing NoOp on healthy ticks while the rule-based agent triggers unnecessary rebalancing which introduces handover overhead.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Reward:</span> +0.2 per cell below 70% load, −0.5 at warning (70–90%), −1.0 at critical (90%+), plus bonus for balanced load distribution.
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            PPO · MlpPolicy 2×64 · Gymnasium · Discrete(4) action space
          </p>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

        {/* Summary stats — 2×2 on mobile, 4×1 on md */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}
          className="ab-stat-grid">
          {[
            { label: 'PPO Avg Reward',  value: ppoAvg.toFixed(3),  color: 'var(--purple)' },
            { label: 'Rule Avg Reward', value: rbAvg.toFixed(3),   color: 'var(--amber)'  },
            { label: 'PPO Win Rate',    value: `${winRate}%`,       color: 'var(--green)'  },
            { label: 'Ticks Compared',  value: totalTicks,          color: 'var(--text-primary)' },
          ].map(({ label, value, color }) => (
            <div key={label} className="stat-mini" style={{ padding: '10px', textAlign: 'center' }}>
              <p style={{ fontSize: '10px', color: 'var(--text-muted)', margin: '0 0 4px 0' }}>{label}</p>
              <p style={{ fontSize: '18px', fontWeight: 700, color, margin: 0, fontFamily: 'monospace' }}>{value}</p>
            </div>
          ))}
        </div>

        {/* Current action row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
          <div style={{
            borderRadius: '10px',
            padding: '12px',
            background: 'var(--accent-dim)',
            border: '1px solid var(--accent-glow)',
          }}>
            <p style={{ fontSize: '10px', color: 'var(--text-muted)', margin: '0 0 4px 0', fontFamily: 'monospace' }}>
              PPO NOW
            </p>
            <p style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent)', margin: 0, wordBreak: 'break-word' }}>
              {ppoActionNow}
            </p>
          </div>
          <div style={{
            borderRadius: '10px',
            padding: '12px',
            background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.2)',
          }}>
            <p style={{ fontSize: '10px', color: 'var(--text-muted)', margin: '0 0 4px 0', fontFamily: 'monospace' }}>
              RULE-BASED NOW
            </p>
            <p style={{ fontSize: '15px', fontWeight: 700, color: 'var(--amber)', margin: 0, wordBreak: 'break-word' }}>
              {rbActionNow}
            </p>
          </div>
        </div>

        {/* Reward time-series */}
        <div className="chart-card" style={{ padding: '14px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            REWARD PER TICK — LAST 60 TICKS
          </p>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis
                dataKey="tick"
                tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                tickLine={false}
                axisLine={false}
                interval={9}
              />
              <YAxis
                domain={[-1, 1]}
                tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
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
                formatter={(v, name) => [v.toFixed(3), name === 'ppo' ? 'PPO Agent' : 'Rule-Based']}
              />
              <Legend
                formatter={(val) => val === 'ppo' ? 'PPO Agent' : 'Rule-Based'}
                wrapperStyle={{ fontSize: 11, color: 'var(--text-secondary)' }}
              />
              <ReferenceLine y={0} stroke="var(--border-accent)" strokeDasharray="3 2" />
              <Line type="monotone" dataKey="ppo" stroke="var(--purple)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="rb"  stroke="var(--amber)"  strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Action distribution */}
        <div className="chart-card" style={{ padding: '14px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            ACTION DISTRIBUTION — LAST 60 TICKS
          </p>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={actionData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis
                dataKey="action"
                tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '8px',
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="PPO"  fill="var(--purple)" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Rule" fill="var(--amber)"  radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '10px', marginTop: '6px', color: 'var(--text-muted)', margin: '6px 0 0 0' }}>
            Both agents choose No-Op when network is healthy. Divergence appears during congestion.
          </p>
        </div>

      </div>

      <style>{`
        @media (min-width: 768px) {
          .ab-stat-grid {
            grid-template-columns: repeat(4, 1fr) !important;
          }
        }
      `}</style>
    </PanelWrapper>
  );
}