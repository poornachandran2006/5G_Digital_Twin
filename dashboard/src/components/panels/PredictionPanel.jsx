import { useSim } from '../../context/SimContext';
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';
import { CELL_COLORS } from '../../constants';

const probColor = (p) => (p > 0.7 ? '#ef4444' : p > 0.4 ? '#f59e0b' : '#10b981');

export default function PredictionPanel() {
  const { state } = useSim();
  const ticks = state.ticks.slice(-60);
  return (
    <div className="space-y-4">
      <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">
        LSTM Congestion Predictions — 30s Ahead
      </h2>
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
  );
}
