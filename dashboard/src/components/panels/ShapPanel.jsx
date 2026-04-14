import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell
} from "recharts";

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
    <div className="bg-gray-900 rounded-xl p-4 space-y-3">

      {/* Header row */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-white font-semibold text-base">
            🔍 Feature Importance
          </h2>
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
  );
}