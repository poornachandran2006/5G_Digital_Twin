import { useSim } from '../../context/SimContext';
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';
import { useState } from 'react';
import InfoModal from '../InfoModal';
import PanelWrapper from '../PanelWrapper';


const SEVERITY_COLOR = {
  normal: '#22c55e',    // green
  warning: '#f59e0b',   // amber
  critical: '#ef4444',  // red
};

const SEVERITY_BG = {
  normal: 'bg-green-900/30 border-green-500/40',
  warning: 'bg-amber-900/30 border-amber-500/40',
  critical: 'bg-red-900/30 border-red-500/40',
};

export default function AnomalyPanel() {
  const [showInfo, setShowInfo] = useState(false);
  const { state } = useSim();
  const { ticks, currentTick } = state;

  // Current anomaly state from latest tick
  const anomaly = currentTick?.anomaly ?? {
    anomaly_score: 0,
    is_anomaly: false,
    severity: 'normal',
  };

  const severity = anomaly.severity || 'normal';
  const score = anomaly.anomaly_score ?? 0;

  // Build chart data from last 60 ticks
  const chartData = ticks.slice(-60).map((t) => ({
    tick: t.tick,
    score: t.anomaly?.anomaly_score ?? 0,
    anomaly: t.anomaly?.is_anomaly ? 1 : 0,
  }));

  // Count anomalies in last 60 ticks
  const recentAnomalyCount = chartData.filter((d) => d.anomaly === 1).length;

  // Score gauge percentage (0–100)
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
      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-bold text-white">Anomaly Detection</h2>
          <button
            onClick={() => setShowInfo(true)}
            className="text-gray-500 hover:text-green-400 transition-colors text-sm"
            title="What is this panel?"
          >
            ⓘ
          </button>
        </div>

        {showInfo && (
          <InfoModal title="Anomaly Detection Panel" onClose={() => setShowInfo(false)}>
            <p>
              This panel uses <span className="text-green-400 font-semibold">Isolation Forest</span> — an unsupervised ML algorithm — to detect abnormal network behaviour without needing labelled examples of failures.
            </p>
            <p>
              <span className="text-white font-semibold">How it works:</span> The model was trained on 10,800 ticks of normal network data. It learns what "normal" looks like. Anything that deviates significantly gets a high anomaly score.
            </p>
            <p>
              <span className="text-white font-semibold">Score thresholds:</span> Below 0.55 = normal, 0.55–0.75 = warning (unusual pattern), above 0.75 = critical (genuine anomaly).
            </p>
            <p>
              This is different from the LSTM predictor — LSTM predicts congestion from trends, Isolation Forest detects unexpected patterns that don't fit any known behaviour.
            </p>
            <p className="text-gray-500 text-xs pt-1">
              "Last 60 ticks flagged" counts how many of the last 60 seconds had an anomaly score above the critical threshold.
            </p>
          </InfoModal>
        )}
        <p className="text-xs text-slate-400">
          Isolation Forest (unsupervised) — trained on 10,800 KPI ticks. Detects statistically
          abnormal network behaviour without labelled data.
        </p>

        {/* Status card */}
        <div className={`rounded-lg border p-4 flex items-center justify-between ${SEVERITY_BG[severity]}`}>
          <div>
            <p className="text-xs text-slate-400 mb-1">Current Status</p>
            <p
              className="text-2xl font-bold uppercase tracking-wide"
              style={{ color: SEVERITY_COLOR[severity] }}
            >
              {severity}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Tick #{currentTick?.tick ?? '—'}
            </p>
          </div>

          {/* Score gauge */}
          <div className="flex flex-col items-center">
            <svg width="80" height="80" viewBox="0 0 80 80">
              {/* Background circle */}
              <circle cx="40" cy="40" r="30" fill="none" stroke="#1e293b" strokeWidth="10" />
              {/* Score arc */}
              <circle
                cx="40"
                cy="40"
                r="30"
                fill="none"
                stroke={SEVERITY_COLOR[severity]}
                strokeWidth="10"
                strokeDasharray={`${gaugePercent * 1.885} 188.5`}
                strokeLinecap="round"
                transform="rotate(-90 40 40)"
              />
              <text x="40" y="44" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                {gaugePercent}%
              </text>
            </svg>
            <p className="text-xs text-slate-400 mt-1">Anomaly Score</p>
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">Score</p>
            <p className="text-xl font-bold text-white">{score.toFixed(3)}</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">Is Anomaly</p>
            <p
              className="text-xl font-bold"
              style={{ color: anomaly.is_anomaly ? '#ef4444' : '#22c55e' }}
            >
              {anomaly.is_anomaly ? 'YES' : 'NO'}
            </p>
          </div>
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-400">Last 60 Ticks</p>
            <p className="text-xl font-bold text-amber-400">{recentAnomalyCount} flagged</p>
          </div>
        </div>

        {/* Score time-series chart */}
        <div>
          <p className="text-xs text-slate-400 mb-2">Anomaly Score — Last 60 Ticks</p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="tick" tick={{ fontSize: 10, fill: '#94a3b8' }} interval={9} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', fontSize: 12 }}
                formatter={(v) => [v.toFixed(3), 'Score']}
              />
              {/* Warning threshold line */}
              <ReferenceLine y={0.55} stroke="#f59e0b" strokeDasharray="4 2" />
              {/* Critical threshold line */}
              <ReferenceLine y={0.75} stroke="#ef4444" strokeDasharray="4 2" />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#818cf8"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-500 mt-1">
            Dashed lines: amber = warning (0.55), red = critical (0.75)
          </p>
        </div>
      </div>
    </PanelWrapper>
  );
}