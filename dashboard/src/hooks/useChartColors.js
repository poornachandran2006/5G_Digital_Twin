import { useMemo } from 'react';

export function useChartColors() {
  return useMemo(() => {
    const s = getComputedStyle(document.documentElement);
    const get = (v) => s.getPropertyValue(v).trim();
    return {
      accent:     get('--accent'),
      purple:     get('--purple'),
      green:      get('--green'),
      amber:      get('--amber'),
      red:        get('--red'),
      textMuted:  get('--text-muted'),
      bgCard:     get('--bg-card'),
      border:     get('--border'),
      chartGrid:  get('--chart-grid'),
      fill0:      get('--chart-fill-0'),
      fill1:      get('--chart-fill-1'),
      fill2:      get('--chart-fill-2'),
      // Cell colors (consistent with constants.js)
      cell: ['#00d4ff', '#8b5cf6', '#10b981'],
      cellLight: ['#0284c7', '#7c3aed', '#059669'],
    };
  }, []);
}