import { useState, useEffect, useRef } from "react";

// ── Animated counter hook ─────────────────────────────────────────────────
function useCounter(target, duration = 1500, suffix = "") {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let start = 0;
    const isFloat = target % 1 !== 0;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(isFloat ? parseFloat(start.toFixed(1)) : Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);
  return count + suffix;
}

// ── Stat card with animated counter ──────────────────────────────────────
function StatCard({ icon, target, suffix, label, delay }) {
  const [visible, setVisible] = useState(false);
  const ref = useRef();
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(t);
  }, [delay]);
  const display = useCounter(visible ? target : 0, 1500, suffix);
  return (
    <div
      ref={ref}
      className="relative group"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(20px)",
        transition: "opacity 0.6s ease, transform 0.6s ease",
        transitionDelay: `${delay}ms`,
      }}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-2xl p-6 flex flex-col items-center text-center
                    group-hover:border-green-500/60 transition-all duration-300"
        style={{
          boxShadow: "0 0 0 0 rgba(74,222,128,0)",
          transition: "box-shadow 0.3s ease, border-color 0.3s ease",
        }}
        onMouseEnter={e => {
          e.currentTarget.style.boxShadow = "0 0 24px 2px rgba(74,222,128,0.15)";
        }}
        onMouseLeave={e => {
          e.currentTarget.style.boxShadow = "0 0 0 0 rgba(74,222,128,0)";
        }}
      >
        <span className="text-4xl mb-3">{icon}</span>
        <span className="text-3xl font-black text-green-400 tabular-nums">{display}</span>
        <span className="text-gray-400 text-xs mt-2 leading-tight">{label}</span>
      </div>
    </div>
  );
}

// ── Fade-in wrapper ───────────────────────────────────────────────────────
function FadeIn({ children, delay = 0, className = "" }) {
  const [visible, setVisible] = useState(false);
  const ref = useRef();
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.1 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(30px)",
        transition: `opacity 0.7s ease ${delay}ms, transform 0.7s ease ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}

// ── Data ──────────────────────────────────────────────────────────────────
const stats = [
  { icon: "📡", target: 3,    suffix: "",   label: "Cell Towers (gNBs)",       delay: 200 },
  { icon: "📱", target: 20,   suffix: "",   label: "Mobile Users (UEs)",        delay: 350 },
  { icon: "⚡", target: 30,   suffix: "s",  label: "Early Warning Window",      delay: 500 },
  { icon: "🎯", target: 98.4, suffix: "%",  label: "Prediction Accuracy (AUC)", delay: 650 },
];

const whatCards = [
  {
    icon: "🏙️",
    title: "The Simulation",
    tag: "SimPy + NumPy",
    body: "A virtual 1km × 1km city with 3 cell towers and 20 phones moving around. Every second, real radio signal strength (SINR) is computed using the same 3GPP path loss formula real 5G networks use.",
    color: "from-blue-500/10 to-transparent",
    border: "border-blue-500/20 hover:border-blue-400/50",
    glow: "rgba(59,130,246,0.15)",
  },
  {
    icon: "🧠",
    title: "The AI Prediction",
    tag: "LSTM + XGBoost · AUC 0.984",
    body: "An LSTM neural network watches 10 seconds of history and predicts tower overload 30 seconds ahead. Paired with XGBoost in a weighted ensemble — zero false alarms in testing.",
    color: "from-green-500/10 to-transparent",
    border: "border-green-500/20 hover:border-green-400/50",
    glow: "rgba(74,222,128,0.15)",
  },
  {
    icon: "🤖",
    title: "The Smart Agent",
    tag: "PPO · 200K training steps",
    body: "A reinforcement learning agent trained for 200,000 steps automatically redistributes users across towers every second — outperforming hand-coded rules in live A/B tests.",
    color: "from-purple-500/10 to-transparent",
    border: "border-purple-500/20 hover:border-purple-400/50",
    glow: "rgba(168,85,247,0.15)",
  },
];

const panelCards = [
  { icon: "🗺️", title: "Network Map",          body: "20 users moving in real time, colored by traffic type (Video/Gaming/IoT/VoIP). Lines show which tower each phone is connected to." },
  { icon: "📊", title: "KPI Charts",            body: "Live speed (Mbps), delay (ms), and load (%) per tower — the exact metrics network engineers track in a real NOC." },
  { icon: "🔮", title: "AI Predictions",        body: "Congestion probability (0–1) per tower, 30 seconds into the future. Above 0.7 triggers a warning." },
  { icon: "🧩", title: "SHAP Explainability",   body: "Shows WHY the AI made each prediction — which inputs pushed the probability up or down. Makes the model transparent." },
  { icon: "🚨", title: "Anomaly Detection",     body: "IsolationForest flags unusual patterns the supervised model wasn't trained on — like hardware faults or sudden traffic spikes." },
  { icon: "⚖️", title: "AI vs Rule-Based A/B", body: "Every tick, both the PPO agent and a simple rule-based agent act. Their rewards are compared live — proving AI beats hand-coded logic." },
];

const techBadges = [
  "Python · SimPy", "PyTorch LSTM", "XGBoost", "PPO · Stable-Baselines3",
  "FastAPI · WebSocket", "React 18 · D3.js", "Prometheus", "Docker · AWS EC2",
];

// ── Main Component ────────────────────────────────────────────────────────
export default function LandingPage({ onEnter }) {
  const [btnHovered, setBtnHovered] = useState(false);

  return (
    <div
      className="min-h-screen text-white overflow-y-auto"
      style={{
        background: "radial-gradient(ellipse at 20% 50%, rgba(17,34,64,0.8) 0%, #050810 60%), #050810",
      }}
    >
      {/* ── Grid pattern overlay ── */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(74,222,128,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(74,222,128,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />

      {/* ── Glow orbs ── */}
      <div
        className="fixed top-0 left-1/4 w-96 h-96 rounded-full pointer-events-none"
        style={{
          background: "radial-gradient(circle, rgba(74,222,128,0.06) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
      />
      <div
        className="fixed bottom-1/4 right-1/4 w-80 h-80 rounded-full pointer-events-none"
        style={{
          background: "radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
      />

      {/* ══════════════════════════════════════════════
          HERO
      ══════════════════════════════════════════════ */}
      <div className="relative flex flex-col items-center justify-center pt-20 pb-16 px-6 text-center">

        {/* Live badge */}
        <div
          className="inline-flex items-center gap-2 border text-green-400 text-sm px-5 py-1.5 rounded-full mb-8"
          style={{
            background: "rgba(74,222,128,0.05)",
            borderColor: "rgba(74,222,128,0.3)",
            boxShadow: "0 0 20px rgba(74,222,128,0.1)",
          }}
        >
          <span
            className="w-2 h-2 rounded-full bg-green-400"
            style={{ boxShadow: "0 0 6px rgba(74,222,128,0.8)", animation: "pulse 2s infinite" }}
          />
          Simulation Running Live
        </div>

        {/* Heading */}
        <h1
          className="text-6xl md:text-7xl font-black tracking-tight mb-5 leading-none"
          style={{ textShadow: "0 0 80px rgba(74,222,128,0.15)" }}
        >
          5G Network
          <br />
          <span
            style={{
              background: "linear-gradient(135deg, #4ade80 0%, #22d3ee 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Digital Twin
          </span>
        </h1>

        {/* Subheading */}
        <p className="text-gray-400 text-lg md:text-xl max-w-2xl mb-12 leading-relaxed">
          A live software simulation of a real urban 5G cellular network — with AI
          that predicts tower congestion{" "}
          <span className="text-white font-semibold">30 seconds before it happens</span>{" "}
          and a reinforcement learning agent that fixes it automatically.
        </p>

        {/* Stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12 w-full max-w-3xl">
          {stats.map((s) => (
            <StatCard key={s.label} {...s} />
          ))}
        </div>

        {/* CTA Button */}
        <button
          onClick={onEnter}
          onMouseEnter={() => setBtnHovered(true)}
          onMouseLeave={() => setBtnHovered(false)}
          className="relative px-12 py-4 rounded-xl text-lg font-bold transition-all duration-300 overflow-hidden"
          style={{
            background: btnHovered
              ? "linear-gradient(135deg, #4ade80, #22d3ee)"
              : "rgba(74,222,128,0.1)",
            border: "1px solid rgba(74,222,128,0.5)",
            color: btnHovered ? "#050810" : "#4ade80",
            boxShadow: btnHovered
              ? "0 0 40px rgba(74,222,128,0.4), 0 0 80px rgba(74,222,128,0.2)"
              : "0 0 20px rgba(74,222,128,0.1)",
            transform: btnHovered ? "scale(1.05)" : "scale(1)",
          }}
        >
          Launch Live Dashboard →
        </button>

        {/* Tech badges */}
        <div className="flex flex-wrap justify-center gap-2 mt-10 max-w-2xl">
          {techBadges.map((b) => (
            <span
              key={b}
              className="text-xs px-3 py-1 rounded-full text-gray-400"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              {b}
            </span>
          ))}
        </div>
      </div>

      {/* ══════════════════════════════════════════════
          WHAT IS THIS
      ══════════════════════════════════════════════ */}
      <div className="max-w-5xl mx-auto px-6 py-16">
        <FadeIn className="text-center mb-12">
          <span className="text-green-400 text-sm font-semibold tracking-widest uppercase">
            How it works
          </span>
          <h2 className="text-3xl font-bold mt-2 mb-3">What is this project?</h2>
          <p className="text-gray-400 text-sm max-w-xl mx-auto">
            No hardware. No real towers. Entirely in software — but using real physics,
            real ML, and real RL.
          </p>
        </FadeIn>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {whatCards.map((c, i) => (
            <FadeIn key={c.title} delay={i * 120}>
              <div
                className={`relative rounded-2xl p-6 border transition-all duration-300 h-full ${c.border}`}
                style={{
                  background: `linear-gradient(135deg, ${c.color.replace("from-", "").replace("/10", "").replace(" to-transparent", "")}1a 0%, rgba(5,8,16,0.8) 100%)`,
                  backdropFilter: "blur(10px)",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.boxShadow = `0 0 30px ${c.glow}`;
                  e.currentTarget.style.transform = "translateY(-4px)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.boxShadow = "none";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                <div className="text-4xl mb-4">{c.icon}</div>
                <div className="text-xs text-gray-500 font-mono mb-2 border border-gray-700 inline-block px-2 py-0.5 rounded">
                  {c.tag}
                </div>
                <h3 className="text-lg font-bold text-white mb-3">{c.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{c.body}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>

      {/* ══════════════════════════════════════════════
          WHAT YOU'LL SEE
      ══════════════════════════════════════════════ */}
      <div
        className="py-16 px-6"
        style={{ background: "rgba(255,255,255,0.015)", borderTop: "1px solid rgba(255,255,255,0.06)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}
      >
        <div className="max-w-5xl mx-auto">
          <FadeIn className="text-center mb-12">
            <span className="text-cyan-400 text-sm font-semibold tracking-widest uppercase">
              Dashboard
            </span>
            <h2 className="text-3xl font-bold mt-2 mb-3">What you'll see inside</h2>
            <p className="text-gray-400 text-sm">
              8 live panels — each showing a different layer of the network's intelligence.
            </p>
          </FadeIn>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {panelCards.map((c, i) => (
              <FadeIn key={c.title} delay={i * 80}>
                <div
                  className="rounded-xl p-5 border border-gray-800 hover:border-gray-600 transition-all duration-300 cursor-default"
                  style={{ background: "rgba(255,255,255,0.02)" }}
                  onMouseEnter={e => {
                    e.currentTarget.style.background = "rgba(255,255,255,0.05)";
                    e.currentTarget.style.transform = "translateY(-2px)";
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xl">{c.icon}</span>
                    <h3 className="font-semibold text-white text-sm">{c.title}</h3>
                  </div>
                  <p className="text-gray-500 text-xs leading-relaxed">{c.body}</p>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════
          FOOTER
      ══════════════════════════════════════════════ */}
      <div className="text-center py-14 px-6">
        <FadeIn>
          <div
            className="inline-flex items-center gap-3 rounded-2xl px-8 py-5 mb-8 border"
            style={{
              background: "rgba(74,222,128,0.04)",
              borderColor: "rgba(74,222,128,0.15)",
            }}
          >
            <span className="text-2xl">👨‍💻</span>
            <div className="text-left">
              <p className="text-white font-bold text-sm text-center mr-5 py-1">Poornachandran</p>
              <p className="text-gray-400 text-xs"> AI Engineer & Full Stack Developer · 3rd year Engineering Student</p>
            </div>
          </div>

          <div className="flex justify-center gap-4">
            <a
              href="https://github.com/poornachandran2006/5G_Digital_Twin"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-gray-400 hover:text-white text-sm px-5 py-2.5 rounded-lg transition-all duration-200"
              style={{ border: "1px solid rgba(255,255,255,0.1)" }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.3)";
                e.currentTarget.style.background = "rgba(255,255,255,0.05)";
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)";
                e.currentTarget.style.background = "transparent";
              }}
            >
              ⭐ View on GitHub
            </a>

            <button
              onClick={onEnter}
              className="inline-flex items-center gap-2 text-green-400 text-sm px-5 py-2.5 rounded-lg transition-all duration-200"
              style={{
                border: "1px solid rgba(74,222,128,0.3)",
                background: "rgba(74,222,128,0.05)",
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = "rgba(74,222,128,0.1)";
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = "rgba(74,222,128,0.05)";
              }}
            >
              Launch Dashboard →
            </button>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}