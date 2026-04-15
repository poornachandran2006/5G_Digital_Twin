import { createContext, useContext, useReducer } from 'react';

const SimContext = createContext(null);

const initialState = {
  ticks: [],
  currentTick: null,
  connected: false,
  reconnecting: false,
  replayMode: false,   // true while a replay is streaming
};

function reducer(state, action) {
  switch (action.type) {
    case 'LOAD_HISTORY': {
      const ticks = action.payload.slice(-300);
      return { ...state, ticks, currentTick: ticks[ticks.length - 1] || null };
    }
    case 'NEW_TICK': {
      const ticks = [...state.ticks, action.payload].slice(-300);
      return {
        ...state,
        ticks,
        currentTick: action.payload,
        replayMode: action.payload?.replay === true,
      };
    }
    case 'SET_CONNECTED':
      return { ...state, connected: action.payload, reconnecting: false };
    case 'SET_RECONNECTING':
      return { ...state, reconnecting: true, connected: false };
    case 'REPLAY_END':
      return { ...state, replayMode: false };
    default:
      return state;
  }
}

export function SimProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return <SimContext.Provider value={{ state, dispatch }}>{children}</SimContext.Provider>;
}

export const useSim = () => useContext(SimContext);