import { useSim } from '../../context/SimContext';

export default function LiveIndicator() {
  const { state } = useSim();
  const color = state.connected ? 'bg-green-400' : state.reconnecting ? 'bg-yellow-400' : 'bg-red-500';
  const label = state.connected ? 'LIVE' : state.reconnecting ? 'RECONNECTING' : 'OFFLINE';
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2.5 h-2.5 rounded-full animate-pulse ${color}`} />
      <span className="text-xs font-mono text-gray-400">{label}</span>
    </div>
  );
}
