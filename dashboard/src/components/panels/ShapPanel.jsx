import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell
} from 'recharts';
import PanelWrapper from '../PanelWrapper';

const POLL_MS = 3000;

const formatFeature = (name) =>
  name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-accent)',
      borderRadius: 8,
      padding: '8px 12px',
      fontSize: 12,
    }}>
      <p style={{ color: 'var(--text-primary)', marginBottom: 4, fontWeight: 600 }}>
        {formatFeature(d.feature)}
      </p>
      <p style={{ color: d.shap_value >= 0 ? 'var(--green)' : 'var(--red)' }}>
        Importance: {(d.shap_value * 100).toFixed(2)}%
      </p>
      <p style={{ color: 'var(--text-secondary)' }}>
        Current value: {d.feature_value}
      </p>
    </div>
  );
};

export default function ShapPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/shap/explanation');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const iv = setInterval(fetchData, POLL_MS);
    return () => clearInterval(iv);
  }, []);

  if (loading) return (
    <div className="chart-card animate-pulse" style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
      Loading feature importance...
    </div>
  );

  if (error) return (
    <div className="chart-card" style={{ color: 'var(--red)', fontSize: '14px' }}>
      Error: {error}
    </div>
  );

  const top10 = (data?.features ?? []).slice(0, 10);

  return (
    <PanelWrapper
      title="SHAP Feature Importance"
      icon="⬢"
      description="Why did the AI predict congestion? SHAP values decompose each XGBoost prediction into exact contributions from all 18 input features. Green pushes toward congestion, red pulls away."
      hint="Longer bars = stronger influence on the prediction"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            SHAP stands for <span style={{ color: 'var(--accent)', fontWeight: 600 }}>SHapley Additive exPlanations</span> — a mathematically rigorous method (from cooperative game theory) to assign credit to each feature for a model's prediction.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            Unlike feature importance by frequency, SHAP values are <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>exact and additive</span>: you can sum all SHAP values for a prediction and get back the model's output minus the base rate.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Base rate</span> = the model's average prediction across training data (≈12% congestion rate). Each bar shows how much that feature shifted the prediction up or down from this base.
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            XGBoost gain-based SHAP · 18 features · refreshes every 3s
          </p>
        </>
      }
    >
      <div className="chart-card space-y-4">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
            Why the model flags congestion — Tick #{data?.tick ?? '—'}
          </p>
          <div className="text-right">
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Base rate</p>
            <p className="font-mono text-sm font-semibold" style={{ color: 'var(--amber)' }}>
              {((data?.base_value ?? 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Bar chart */}
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={top10}
            layout="vertical"
            margin={{ top: 4, right: 24, left: 150, bottom: 4 }}
          >
            <XAxis
              type="number"
              domain={[-1, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={145}
              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={formatFeature}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="var(--border-accent)" strokeWidth={1} />
            <Bar dataKey="shap_value" radius={[0, 3, 3, 0]} maxBarSize={18}>
              {top10.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.shap_value >= 0 ? '#34d399' : '#f87171'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div
          className="flex gap-5 text-xs pt-3"
          style={{ borderTop: '1px solid var(--border)', color: 'var(--text-secondary)' }}
        >
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm inline-block" style={{ background: '#34d399' }} />
            Increases congestion risk
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm inline-block" style={{ background: '#f87171' }} />
            Decreases congestion risk
          </span>
          <span className="ml-auto italic" style={{ color: 'var(--text-muted)' }}>
            XGBoost gain · refreshes every 3s
          </span>
        </div>
      </div>
    </PanelWrapper>
  );
}