import { useEffect, useState } from "react";
import InfoModal from "../InfoModal";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell
} from "recharts";
import PanelWrapper from '../PanelWrapper';

const POLL_INTERVAL_MS = 3000;

const formatFeature = (name) =>
  name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "#1f2937",
      border: "1px solid #374151",
      borderRadius: 8,
      padding: "8px 12px",
      fontSize: 12,
    }}>
      <p style={{ color: "#f9fafb", marginBottom: 4, fontWeight: 600 }}>
        {formatFeature(d.feature)}
      </p>
      <p style={{ color: d.shap_value >= 0 ? "#34d399" : "#f87171" }}>
        Importance: {(d.shap_value * 100).toFixed(2)}%
      </p>
      <p style={{ color: "#9ca3af" }}>
        Current value: {d.feature_value}
      </p>
    </div>
  );
};

export default function ShapPanel() {
  const [showInfo, setShowInfo] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/shap/explanation");
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
    const interval = setInterval(fetchData, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  if (loading) return (
    <div className="bg-gray-900 rounded-xl p-4 text-gray-400 text-sm animate-pulse">
      Loading feature importance...
    </div>
  );

  if (error) return (
    <div className="bg-gray-900 rounded-xl p-4 text-red-400 text-sm">
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
            Unlike feature importance by frequency, SHAP values are <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>exact and additive</span>: you can sum all SHAP values for a prediction and get back the model's output minus the base rate. No approximations.
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
      <div className="bg-gray-900 rounded-xl p-4 space-y-3">

        {/* Header row */}
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-white font-semibold text-base">
                🔍 Feature Importance
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
              <InfoModal title="SHAP Feature Importance" onClose={() => setShowInfo(false)}>
                <p>
                  This panel answers: <span className="text-green-400 font-semibold">why did the AI predict congestion?</span> It uses SHAP (SHapley Additive exPlanations) to break down every prediction into contributions from individual features.
                </p>
                <p>
                  <span className="text-white font-semibold">Green bars</span> = that feature is pushing the prediction toward congestion. <span className="text-red-400 font-semibold">Red bars</span> = that feature is pulling it away from congestion.
                </p>
                <p>
                  <span className="text-white font-semibold">Base rate</span> is the model's default prediction if it knew nothing — the average congestion rate across training data.
                </p>
                <p>
                  The underlying model is XGBoost (gradient-boosted decision trees). SHAP values come from the exact mathematical Shapley values — not approximations.
                </p>
                <p className="text-gray-500 text-xs pt-1">
                  Refreshes every 3 seconds. Top 10 features shown by absolute importance.
                </p>
              </InfoModal>
            )}
            <p className="text-gray-400 text-xs mt-0.5">
              Why the model flags congestion — Tick #{data?.tick ?? "—"}
            </p>
          </div>
          <div className="text-right">
            <p className="text-gray-500 text-xs">Base rate</p>
            <p className="text-yellow-400 font-mono text-sm font-semibold">
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
              tick={{ fill: "#9ca3af", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#374151" }}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={145}
              tick={{ fill: "#d1d5db", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={formatFeature}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="#4b5563" strokeWidth={1} />
            <Bar dataKey="shap_value" radius={[0, 3, 3, 0]} maxBarSize={18}>
              {top10.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.shap_value >= 0 ? "#34d399" : "#f87171"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="flex gap-5 text-xs text-gray-400 pt-2 border-t border-gray-800">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm bg-green-400 inline-block" />
            Increases congestion risk
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm bg-red-400 inline-block" />
            Decreases congestion risk
          </span>
          <span className="ml-auto text-gray-600 italic">
            XGBoost gain · refreshes every 3s
          </span>
        </div>

      </div>
    </PanelWrapper>
  );
}