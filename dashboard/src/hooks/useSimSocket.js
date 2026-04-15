import { useEffect, useRef } from 'react';
import { useSim } from '../context/SimContext';

export function useSimSocket() {
  const { dispatch } = useSim();
  const wsRef = useRef(null);
  const timerRef = useRef(null);

  function connect() {
    const ws = new WebSocket('ws://localhost:8000/ws/simulation');
    wsRef.current = ws;
    ws.onopen = () => {
      dispatch({ type: 'SET_CONNECTED', payload: true });
      clearTimeout(timerRef.current);
    };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'history') dispatch({ type: 'LOAD_HISTORY', payload: msg.payload.ticks });
      else if (msg.type === 'tick_update') dispatch({ type: 'NEW_TICK', payload: msg.payload });
      else if (msg.type === 'replay_end') dispatch({ type: 'REPLAY_END' });
    };
    ws.onclose = () => {
      dispatch({ type: 'SET_RECONNECTING' });
      timerRef.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
  }

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      clearTimeout(timerRef.current);
    };
  }, []);
}
