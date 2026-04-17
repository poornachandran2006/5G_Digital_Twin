import { useSim } from '../../context/SimContext';
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';
import PanelWrapper from '../PanelWrapper';

const SEVERITY_COLOR = {
  normal:   'var(--green)',
  warning:  'var(--amber)',
  critical: 'var(--red)',
};

const SEVERITY_BORDER = {
  normal:   'var(--green)',
  warning:  'var(--amber)',
  critical: 'var(--red)',
};

export default function AnomalyPanel() {
  const { state } = useSim();
  const { ticks, currentTick } = state;

  const anomaly = currentTick?.anomaly ?? { anomaly_score: 0, is_anomaly: false, severity: 'normal' };
  const severity = anomaly.severity || 'normal';
  const score = anomaly.anomaly_score ?? 0;

  const chartData = ticks.slice(-60).map((t) => ({
    tick: t.tick,
    score: t.anomaly?.anomaly_score ?? 0,
    anomaly: t.anomaly?.is_anomaly ? 1 : 0,
  }));

  const recentAnomalyCount = chartData.filter((d) => d.anomaly === 1).length;
  const gaugePercent = Math.round(score * 100);

  return (
    <PanelWrapper
      title="Anomaly Detection"
      icon="⚠"
      description="IsolationForest detects unusual network behaviour without labelled examples. It learns what normal looks like across 10,800 ticks, then flags anything that deviates significantly."
      hint="Score above 0.75 = genuine anomaly (red zone)"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Isolation Forest</span> is an unsupervised algorithm — it requires no labelled failure examples. Instead it learns the normal distribution by randomly partitioning the feature space. Points that require fewer partitions to isolate are anomalies.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            This is a <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>second line of defence</span> beyond the LSTM. The LSTM can only detect congestion patterns it was trained on. Isolation Forest catches unknown failure modes — hardware faults, unusual traffic spikes, coordinated anomalies — that never appeared in training data.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--green)', fontWeight: 600 }}>Normal</span> (below 0.55) · <span style={{ color: 'var(--amber)', fontWeight: 600 }}>Warning</span> (0.55–0.75) · <span style={{ color: 'var(--red)', fontWeight: 600 }}>Critical</span> (above 0.75)
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            100 estimators · contamination = 0.05 · scikit-learn
          </p>
        </>
      }
    >
      <div className="space-y-4">
        {/* Status card */}
        <div
          className="rounded-xl p-4 flex items-center justify-between"
          style={{
            background: 'var(--bg-card)',
            border: `1px solid ${SEVERITY_BORDER[severity]}44`,
            transition: 'var(--transition)',
          }}
        >
          <div>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Current Status</p>
            <p className="text-2xl font-bold uppercase tracking-wide" style={{ color: SEVERITY_COLOR[severity] }}>
              {severity}
            </p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              Tick #{currentTick?.tick ?? '—'}
            </p>
          </div>
          {/* SVG gauge */}
          <div className="flex flex-col items-center">
            <svg width="80" height="80" viewBox="0 0 80 80">
              <circle cx="40" cy="40" r="30" fill="none" stroke="var(--border)" strokeWidth="10" />
              <circle
                cx="40" cy="40" r="30"
                fill="none"
                stroke={SEVERITY_COLOR[severity]}
                strokeWidth="10"
                strokeDasharray={`${gaugePercent * 1.885} 188.5`}
                strokeLinecap="round"
                transform="rotate(-90 40 40)"
              />
              <text x="40" y="44" textAnchor="middle" fill="var(--text-primary)" fontSize="14" fontWeight="bold">
                {gaugePercent}%
              </text>
            </svg>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Anomaly Score</p>
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Score', value: score.toFixed(3), color: 'var(--text-primary)' },
            { label: 'Is Anomaly', value: anomaly.is_anomaly ? 'YES' : 'NO', color: anomaly.is_anomaly ? 'var(--red)' : 'var(--green)' },
            { label: 'Last 60 Ticks', value: `${recentAnomalyCount} flagged`, color: 'var(--amber)' },
          ].map(({ label, value, color }) => (
            <div key={label} className="stat-mini">
              <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{label}</p>
              <p className="text-xl font-bold" style={{ color }}>{value}</p>
            </div>
          ))}
        </div>

        {/* Chart */}
        <div className="chart-card">
          <p className="text-xs font-mono mb-3" style={{ color: 'var(--text-muted)' }}>
            Anomaly Score — Last 60 Ticks
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="tick" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval={9} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '8px',
                  fontSize: 12,
                }}
                formatter={(v) => [v.toFixed(3), 'Score']}
              />
              <ReferenceLine y={0.55} stroke="var(--amber)" strokeDasharray="4 2" />
              <ReferenceLine y={0.75} stroke="var(--red)" strokeDasharray="4 2" />
              <Line
                type="monotone"
                dataKey="score"
                stroke="var(--purple)"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            Dashed lines: amber = warning (0.55), red = critical (0.75)
          </p>
        </div>
      </div>
    </PanelWrapper>
  );
}