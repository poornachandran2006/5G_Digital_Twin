import { useState } from 'react';
import { useSim } from '../../context/SimContext';
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';
import { CELL_COLORS } from '../../constants';
import InfoModal from '../InfoModal';
import PanelWrapper from '../PanelWrapper';

const probColor = (p) => (p > 0.7 ? '#ef4444' : p > 0.4 ? '#f59e0b' : '#10b981');

export default function PredictionPanel() {
  const { state } = useSim();
  const [showInfo, setShowInfo] = useState(false);
  const ticks = state.ticks.slice(-60);

  return (
    <PanelWrapper
      title="Congestion Predictions"
      icon="◈"
      description="LSTM + XGBoost ensemble predicting cell congestion 30 seconds ahead. Probability above 0.5 triggers a warning. The model achieves AUC 0.984 with zero false positives."
      hint="Predictions are 30 ticks (30 seconds) into the future"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            Two models are combined: <span style={{ color: 'var(--accent)', fontWeight: 600 }}>LSTM</span> captures temporal trends (rising load over 10 ticks → likely congestion), and <span style={{ color: 'var(--purple)', fontWeight: 600 }}>XGBoost</span> captures sharp feature interactions (load AND UE count both high simultaneously).
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Ensemble weights:</span> 0.6 × LSTM + 0.4 × XGBoost. The 60/40 split was tuned empirically — LSTM contributes more because temporal drift matters most for early warning.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Precision = 1.0</span> means zero false alarms. Every alert the model raises is a real event. This is the most critical metric for production NOC use — false alarms erode engineer trust.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            18 input features · 10-tick sequence window · trained on 10,800 rows
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            F1: 0.871 · AUC-ROC: 0.984 · Precision: 1.000 · Recall: 0.771
          </p>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">
            LSTM Congestion Predictions — 30s Ahead
          </h2>
          <button
            onClick={() => setShowInfo(true)}
            className="text-gray-500 hover:text-green-400 transition-colors text-sm"
            title="What is this panel?"
          >
            ⓘ
          </button>
        </div>

        {showInfo && (
          <InfoModal title="Congestion Predictions Panel" onClose={() => setShowInfo(false)}>
            <p>
              This panel shows the AI's prediction of whether each tower will become{' '}
              <span className="text-green-400 font-semibold">congested in the next 30 seconds</span> —
              updated every tick.
            </p>
            <p>
              <span className="text-white font-semibold">How the AI works:</span> An LSTM neural
              network reads the last 10 seconds of network data (load, throughput, user count, SINR)
              and outputs a probability from 0 to 100%. LSTM stands for Long Short-Term Memory — it
              remembers trends over time, not just the current snapshot.
            </p>
            <p>
              It's paired with an XGBoost model in a weighted ensemble (60% LSTM + 40% XGBoost).
              The ensemble achieves <span className="text-green-400 font-semibold">AUC = 0.984</span>{' '}
              and <span className="text-green-400 font-semibold">Precision = 1.0</span> — meaning
              every alarm it raises is a real congestion event. Zero false alarms.
            </p>
            <p>
              <span className="text-white font-semibold">Color coding:</span>{' '}
              <span className="text-green-400">Green</span> = healthy (&lt;40%),{' '}
              <span className="text-yellow-400">Yellow</span> = warning (40–70%),{' '}
              <span className="text-red-400">Red</span> = critical (&gt;70%).
            </p>
            <p className="text-gray-500 text-xs pt-1">
              Prediction horizon = 30 ticks = 30 seconds. This gives network operators time to
              act before users notice any slowdown.
            </p>
          </InfoModal>
        )}

        {[0, 1, 2].map((i) => {
          const prob = state.currentTick?.congestion_predictions?.[i] ?? 0;
          const history = ticks.map((t) => ({ v: t.congestion_predictions?.[i] ?? 0 }));
          const color = probColor(prob);
          return (
            <div key={i} className="bg-[#111827] rounded-xl border border-gray-800 p-4">
              <div className="flex justify-between items-center mb-3">
                <span className="text-sm font-mono" style={{ color: CELL_COLORS[i] }}>Cell {i}</span>
                <span className="text-lg font-mono" style={{ color }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-800 rounded-full mb-3">
                <div
                  className="h-2 rounded-full transition-all duration-300"
                  style={{ width: `${prob * 100}%`, background: color }}
                />
              </div>
              <ResponsiveContainer width="100%" height={60}>
                <LineChart data={history}>
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke={CELL_COLORS[i]}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#111827',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      fontSize: '11px',
                    }}
                    formatter={(v) => [`${(v * 100).toFixed(1)}%`]}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </PanelWrapper>
  );
}