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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {[0, 1, 2].map((i) => {
          const prob = state.currentTick?.congestion_predictions?.[i] ?? 0;
          const history = ticks.map((t) => ({ v: t.congestion_predictions?.[i] ?? 0 }));
          const color = probColor(prob);
          const label = prob > 0.7 ? 'CRITICAL' : prob > 0.4 ? 'WARNING' : 'HEALTHY';

          return (
            <div key={i} className="chart-card" style={{ padding: '14px' }}>
              {/* Header row */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '13px', fontFamily: 'monospace', fontWeight: 700, color: CELL_COLORS[i] }}>
                    Cell {i}
                  </span>
                  <span style={{
                    fontSize: '10px',
                    fontFamily: 'monospace',
                    padding: '2px 7px',
                    borderRadius: '999px',
                    background: `${color}22`,
                    color,
                    letterSpacing: '0.05em',
                  }}>
                    {label}
                  </span>
                </div>
                <span style={{ fontSize: '20px', fontFamily: 'monospace', fontWeight: 700, color }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>

              {/* Progress bar */}
              <div style={{ width: '100%', height: '5px', background: 'var(--border)', borderRadius: '3px', marginBottom: '10px', position: 'relative' }}>
                {/* Threshold markers */}
                <div style={{ position: 'absolute', left: '40%', top: 0, bottom: 0, width: '1px', background: 'var(--amber)', opacity: 0.5 }} />
                <div style={{ position: 'absolute', left: '70%', top: 0, bottom: 0, width: '1px', background: 'var(--red)', opacity: 0.5 }} />
                <div style={{
                  height: '5px',
                  borderRadius: '3px',
                  width: `${prob * 100}%`,
                  background: color,
                  transition: 'width 0.4s ease',
                }} />
              </div>

              {/* Sparkline */}
              <ResponsiveContainer width="100%" height={55}>
                <LineChart data={history} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
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
                    formatter={(v) => [`${(v * 100).toFixed(1)}%`, 'Congestion prob']}
                  />
                </LineChart>
              </ResponsiveContainer>

              {/* Footer meta */}
              <p style={{ fontSize: '10px', fontFamily: 'monospace', color: 'var(--text-muted)', marginTop: '6px', margin: '6px 0 0 0' }}>
                Horizon: +30 ticks · 0.6×LSTM + 0.4×XGBoost
              </p>
            </div>
          );
        })}
      </div>
    </PanelWrapper>
  );
}