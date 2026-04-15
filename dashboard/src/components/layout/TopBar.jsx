import { useState } from 'react';
import { useSim } from '../../context/SimContext';
import LiveIndicator from '../shared/LiveIndicator';

export default function TopBar() {
  const { state } = useSim();
  const tick = state.currentTick?.tick ?? 0;
  const [recording, setRecording] = useState(false);
  const [hasBuffer, setHasBuffer] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const [speed, setSpeed] = useState(2);

  const start = () => fetch('/api/simulation/start', { method: 'POST' });
  const stop  = () => fetch('/api/simulation/stop',  { method: 'POST' });

  const startRecord = async () => {
    await fetch('/api/replay/record/start', { method: 'POST' });
    setRecording(true);
    setHasBuffer(false);
  };

  const stopRecord = async () => {
    const res = await fetch('/api/replay/record/stop', { method: 'POST' });
    const data = await res.json();
    setRecording(false);
    setHasBuffer(data.buffer_size > 0);
  };

  const startReplay = async () => {
    await fetch(`/api/replay/play?speed=${speed}`, { method: 'POST' });
    setReplaying(true);
    // Auto-clear replaying flag after estimated duration
    const statusRes = await fetch('/api/replay/status');
    const s = await statusRes.json();
    setTimeout(() => setReplaying(false), (s.buffer_size / speed) * 1000 + 1000);
  };

  const stopReplay = async () => {
    await fetch('/api/replay/stop', { method: 'POST' });
    setReplaying(false);
  };

  return (
    <header className="h-14 bg-[#0d1424] border-b border-gray-800 flex items-center justify-between px-6 shrink-0">
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-mono text-[#00d4ff] tracking-widest uppercase">
          5G Network Digital Twin
        </h1>
        <LiveIndicator />
        {state.replayMode && (
          <span className="text-xs font-mono text-yellow-400 border border-yellow-400/30 px-2 py-0.5 rounded animate-pulse">
            ⏪ REPLAY
          </span>
        )}
      </div>

      <div className="flex items-center gap-6">
        <span className="text-xs font-mono text-gray-400">
          TICK <span className="text-white">{tick}</span>
        </span>

        {/* Replay controls */}
        <div className="flex items-center gap-2 border-l border-gray-700 pl-4">
          <span className="text-xs font-mono text-gray-500">REPLAY</span>

          {!recording ? (
            <button
              type="button"
              onClick={startRecord}
              className="px-2 py-1 text-xs bg-orange-500/20 text-orange-400 border border-orange-500/30 rounded hover:bg-orange-500/30 transition-colors"
            >
              ● REC
            </button>
          ) : (
            <button
              type="button"
              onClick={stopRecord}
              className="px-2 py-1 text-xs bg-orange-500/40 text-orange-300 border border-orange-400/50 rounded animate-pulse"
            >
              ■ STOP REC
            </button>
          )}

          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="text-xs bg-gray-800 text-gray-300 border border-gray-600 rounded px-1 py-1 font-mono"
          >
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
            <option value={8}>8×</option>
          </select>

          {!replaying ? (
            <button
              type="button"
              onClick={startReplay}
              disabled={!hasBuffer}
              className="px-2 py-1 text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded hover:bg-blue-500/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ▶ PLAY
            </button>
          ) : (
            <button
              type="button"
              onClick={stopReplay}
              className="px-2 py-1 text-xs bg-blue-500/40 text-blue-300 border border-blue-400/50 rounded"
            >
              ■ STOP
            </button>
          )}

          {hasBuffer && !replaying && (
            <span className="text-xs font-mono text-gray-500">
              {/* show tick count once recorded */}
            </span>
          )}
        </div>

        {/* Sim controls */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={start}
            className="px-3 py-1 text-xs bg-green-500/20 text-green-400 border border-green-500/30 rounded hover:bg-green-500/30 transition-colors"
          >
            START
          </button>
          <button
            type="button"
            onClick={stop}
            className="px-3 py-1 text-xs bg-red-500/20 text-red-400 border border-red-500/30 rounded hover:bg-red-500/30 transition-colors"
          >
            STOP
          </button>
        </div>
      </div>
    </header>
  );
}