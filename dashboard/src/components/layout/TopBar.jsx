import { useSim } from '../../context/SimContext';
import LiveIndicator from '../shared/LiveIndicator';

export default function TopBar() {
  const { state } = useSim();
  const tick = state.currentTick?.tick ?? 0;
  const start = () => fetch('/api/simulation/start', { method: 'POST' });
  const stop = () => fetch('/api/simulation/stop', { method: 'POST' });
  return (
    <header className="h-14 bg-[#0d1424] border-b border-gray-800 flex items-center justify-between px-6 shrink-0">
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-mono text-[#00d4ff] tracking-widest uppercase">
          5G Network Digital Twin
        </h1>
        <LiveIndicator />
      </div>
      <div className="flex items-center gap-6">
        <span className="text-xs font-mono text-gray-400">
          TICK <span className="text-white">{tick}</span>
        </span>
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
