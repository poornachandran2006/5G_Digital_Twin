import { useEffect, useRef } from 'react';
import { useSim } from '../context/SimContext';

export function useSimSocket() {
  const { dispatch } = useSim();
  const wsRef = useRef(null);
  const timerRef = useRef(null);

  function connect() {
    // Dynamically resolve WebSocket URL so it works locally AND on EC2
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/simulation`;

    const ws = new WebSocket(wsUrl);
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