import { useEffect } from "react";

export default function InfoModal({ title, onClose, children }) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-4"
      style={{ background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
    >
      <div
        className="relative max-w-lg w-full rounded-2xl p-6 text-white"
        style={{
          background: "linear-gradient(135deg, #0f172a 0%, #0a0f1e 100%)",
          border: "1px solid rgba(74,222,128,0.25)",
          boxShadow: "0 0 40px rgba(74,222,128,0.1), 0 20px 60px rgba(0,0,0,0.5)",
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-white">{title}</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl leading-none"
          >
            ✕
          </button>
        </div>

        {/* Divider */}
        <div className="h-px bg-gradient-to-r from-green-500/40 to-transparent mb-4" />

        {/* Content */}
        <div className="text-gray-300 text-sm leading-relaxed space-y-3">
          {children}
        </div>

        {/* Footer hint */}
        <p className="text-gray-600 text-xs mt-5 text-right">Press Esc or click outside to close</p>
      </div>
    </div>
  );
}