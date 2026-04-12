export const CELL_COLORS = { 0: '#00d4ff', 1: '#8b5cf6', 2: '#10b981' };
export const ACTION_NAMES = { 0: 'NoOp', 1: 'LoadBalance', 2: 'PowerCtrl', 3: 'Handover' };
export const STATUS_COLOR = (load) =>
  load > 0.9 ? '#ef4444' : load > 0.7 ? '#f59e0b' : '#10b981';
