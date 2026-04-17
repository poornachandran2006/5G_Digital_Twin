import { useSim } from '../../context/SimContext';
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';
import { CELL_COLORS } from '../../constants';
import PanelWrapper from '../PanelWrapper';

const probColor = (p) => (p > 0.7 ? 'var(--red)' : p > 0.4 ? 'var(--amber)' : 'var(--green)');

export default function PredictionPanel() {
  const { state } = useSim();
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
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            F1: 0.871 · AUC-ROC: 0.984 · Precision: 1.000 · Recall: 0.771
          </p>
        </>
      }
    >
      <div className="space-y-4">
        {[0, 1, 2].map((i) => {
          const prob = state.currentTick?.congestion_predictions?.[i] ?? 0;
          const history = ticks.map((t) => ({ v: t.congestion_predictions?.[i] ?? 0 }));
          const color = probColor(prob);
          return (
            <div key={i} className="chart-card">
              <div className="flex justify-between items-center mb-3">
                <span className="text-sm font-mono" style={{ color: CELL_COLORS[i] }}>Cell {i}</span>
                <span className="text-lg font-mono font-bold" style={{ color }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
              {/* Progress bar */}
              <div
                className="w-full h-2 rounded-full mb-3"
                style={{ background: 'var(--border)' }}
              >
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
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-accent)',
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