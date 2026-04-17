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
      <p style={{ color: 'var(--text-primary)', marginBottom: 4, fontWeight: 600, margin: '0 0 4px 0' }}>
        {formatFeature(d.feature)}
      </p>
      <p style={{ color: d.shap_value >= 0 ? 'var(--green)' : 'var(--red)', margin: '0 0 2px 0' }}>
        Importance: {(d.shap_value * 100).toFixed(2)}%
      </p>
      <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
        Value: {d.feature_value}
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
    <div className="chart-card" style={{ color: 'var(--text-muted)', fontSize: '14px', padding: '20px' }}>
      Loading feature importance...
    </div>
  );

  if (error) return (
    <div className="chart-card" style={{ color: 'var(--red)', fontSize: '14px', padding: '20px' }}>
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
      <div className="chart-card" style={{ padding: '14px' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <p style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)', margin: 0 }}>
            Why the model flags congestion — Tick #{data?.tick ?? '—'}
          </p>
          <div style={{ textAlign: 'right' }}>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: '0 0 2px 0' }}>Base rate</p>
            <p style={{ fontFamily: 'monospace', fontSize: '14px', fontWeight: 600, color: 'var(--amber)', margin: 0 }}>
              {((data?.base_value ?? 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Responsive chart — dynamically adjusts left margin for labels */}
        <div className="shap-chart-wrap">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={top10}
              layout="vertical"
              margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
            >
              <XAxis
                type="number"
                domain={[-1, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: 'var(--border)' }}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={0}
                tick={false}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine x={0} stroke="var(--border-accent)" strokeWidth={1} />
              <Bar dataKey="shap_value" radius={[0, 3, 3, 0]} maxBarSize={20} label={{ position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 10, formatter: (_, entry) => formatFeature(entry?.feature ?? '') }}>
                {top10.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={entry.shap_value >= 0 ? '#34d399' : '#f87171'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature name list — alternative for mobile */}
        <div className="shap-feature-list">
          {top10.map((entry, i) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '5px 0',
              borderBottom: i < top10.length - 1 ? '1px solid var(--border)' : 'none',
            }}>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                {formatFeature(entry.feature)}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: 60, height: 6, background: 'var(--border)', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
                  <div style={{
                    position: 'absolute',
                    top: 0, bottom: 0,
                    left: entry.shap_value >= 0 ? '50%' : `${50 + entry.shap_value * 50}%`,
                    width: `${Math.abs(entry.shap_value) * 50}%`,
                    background: entry.shap_value >= 0 ? '#34d399' : '#f87171',
                    borderRadius: 3,
                  }} />
                </div>
                <span style={{
                  fontSize: '11px',
                  fontFamily: 'monospace',
                  color: entry.shap_value >= 0 ? '#34d399' : '#f87171',
                  minWidth: '40px',
                  textAlign: 'right',
                }}>
                  {(entry.shap_value * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          fontSize: '11px',
          paddingTop: '12px',
          borderTop: '1px solid var(--border)',
          color: 'var(--text-secondary)',
          marginTop: '12px',
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: 12, height: 8, borderRadius: '2px', display: 'inline-block', background: '#34d399' }} />
            Increases congestion risk
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: 12, height: 8, borderRadius: '2px', display: 'inline-block', background: '#f87171' }} />
            Decreases risk
          </span>
          <span style={{ marginLeft: 'auto', fontStyle: 'italic', color: 'var(--text-muted)' }}>
            Refreshes every 3s
          </span>
        </div>
      </div>

      <style>{`
        /* On mobile: hide the recharts chart, show simple list */
        @media (max-width: 480px) {
          .shap-chart-wrap { display: none; }
          .shap-feature-list { display: block !important; }
        }
        /* On larger screens: show chart, hide list */
        @media (min-width: 481px) {
          .shap-chart-wrap { display: block; }
          .shap-feature-list { display: none; }
        }
      `}</style>
    </PanelWrapper>
  );
}