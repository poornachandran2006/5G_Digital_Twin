import { useSim } from '../../context/SimContext';
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';
import PanelWrapper from '../PanelWrapper';

const SEVERITY_COLOR = {
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
  const color = SEVERITY_COLOR[severity];

  // SVG arc for gauge
  const RADIUS = 36;
  const STROKE = 8;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
  const arc = (gaugePercent / 100) * CIRCUMFERENCE;

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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

        {/* Status card — horizontal on all sizes */}
        <div
          style={{
            borderRadius: '12px',
            padding: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            background: 'var(--bg-card)',
            border: `1px solid ${color}44`,
            transition: 'var(--transition)',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: '0 0 4px 0' }}>Current Status</p>
            <p style={{ fontSize: '22px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color, margin: '0 0 4px 0' }}>
              {severity}
            </p>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0 }}>
              Tick #{currentTick?.tick ?? '—'}
            </p>
          </div>

          {/* Circular gauge */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
            <svg width="90" height="90" viewBox="0 0 90 90">
              <circle cx="45" cy="45" r={RADIUS} fill="none" stroke="var(--border)" strokeWidth={STROKE} />
              <circle
                cx="45" cy="45" r={RADIUS}
                fill="none"
                stroke={color}
                strokeWidth={STROKE}
                strokeDasharray={`${arc} ${CIRCUMFERENCE}`}
                strokeLinecap="round"
                transform="rotate(-90 45 45)"
                style={{ transition: 'stroke-dasharray 0.4s ease, stroke 0.4s ease' }}
              />
              <text x="45" y="49" textAnchor="middle" fill="var(--text-primary)" fontSize="15" fontWeight="bold" fontFamily="monospace">
                {gaugePercent}%
              </text>
            </svg>
            <p style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px', fontFamily: 'monospace' }}>
              Anomaly Score
            </p>
          </div>
        </div>

        {/* Stats row — 3 cols, compact on mobile */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
          {[
            { label: 'Raw Score',    value: score.toFixed(3),                  color: 'var(--text-primary)' },
            { label: 'Is Anomaly',   value: anomaly.is_anomaly ? 'YES' : 'NO', color: anomaly.is_anomaly ? 'var(--red)' : 'var(--green)' },
            { label: 'Flagged / 60', value: `${recentAnomalyCount}`,           color: 'var(--amber)' },
          ].map(({ label, value, color: c }) => (
            <div key={label} className="stat-mini" style={{ padding: '10px', textAlign: 'center' }}>
              <p style={{ fontSize: '10px', color: 'var(--text-muted)', margin: '0 0 4px 0' }}>{label}</p>
              <p style={{ fontSize: '18px', fontWeight: 700, color: c, margin: 0, fontFamily: 'monospace' }}>{value}</p>
            </div>
          ))}
        </div>

        {/* Chart */}
        <div className="chart-card" style={{ padding: '14px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            ANOMALY SCORE — LAST 60 TICKS
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
                domain={[0, 1]}
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
                formatter={(v) => [v.toFixed(3), 'Score']}
              />
              <ReferenceLine y={0.55} stroke="var(--amber)" strokeDasharray="4 2" strokeWidth={1} />
              <ReferenceLine y={0.75} stroke="var(--red)" strokeDasharray="4 2" strokeWidth={1} />
              <Line
                type="monotone"
                dataKey="score"
                stroke="var(--purple)"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '10px', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--amber)' }}>― 0.55 warning</span>
            <span style={{ color: 'var(--red)' }}>― 0.75 critical</span>
          </div>
        </div>

      </div>
    </PanelWrapper>
  );
}