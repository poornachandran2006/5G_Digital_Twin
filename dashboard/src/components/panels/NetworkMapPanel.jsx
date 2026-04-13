import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useSim } from '../../context/SimContext';
import { CELL_COLORS } from '../../constants';

// gNB positions are fixed — match simulation/engine.py _GNB_POSITIONS
// engine uses (200,500), (500,200), (800,700) on a 1000x1000 grid
const GNB = [
  { id: 0, x: 200, y: 500 },
  { id: 1, x: 500, y: 200 },
  { id: 2, x: 800, y: 700 },
];

// Coverage radius visual guide (not physics — purely decorative)
const COVERAGE_RADIUS = 280;

export default function NetworkMapPanel() {
  const { state } = useSim();
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const staticRenderedRef = useRef(false);

  // ── Static layer: render gNBs and coverage rings ONCE on mount ──────────
  useEffect(() => {
    if (!svgRef.current || staticRenderedRef.current) return;
    staticRenderedRef.current = true;

    const svg = d3.select(svgRef.current);

    // Coverage rings (behind everything)
    svg.selectAll('.coverage-ring')
      .data(GNB)
      .enter()
      .append('circle')
      .attr('class', 'coverage-ring')
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('r', COVERAGE_RADIUS)
      .attr('fill', 'none')
      .attr('stroke', (d) => CELL_COLORS[d.id])
      .attr('stroke-opacity', 0.08)
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '6 4');

    // gNB towers (triangles)
    svg.selectAll('.gnb-tower')
      .data(GNB)
      .enter()
      .append('polygon')
      .attr('class', 'gnb-tower')
      .attr('points', (d) =>
        `${d.x},${d.y - 20} ${d.x - 15},${d.y + 12} ${d.x + 15},${d.y + 12}`
      )
      .attr('fill', (d) => CELL_COLORS[d.id])
      .attr('fill-opacity', 0.95)
      .attr('filter', 'drop-shadow(0 0 6px rgba(0,200,255,0.4))');

    // gNB labels
    svg.selectAll('.gnb-label')
      .data(GNB)
      .enter()
      .append('text')
      .attr('class', 'gnb-label')
      .attr('x', (d) => d.x)
      .attr('y', (d) => d.y + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', (d) => CELL_COLORS[d.id])
      .attr('font-size', 11)
      .attr('font-family', 'monospace')
      .attr('font-weight', 'bold')
      .text((d) => `gNB-${d.id}`);

  }, []); // empty deps — runs once only

  // ── Dynamic layer: update UE positions and link lines on every tick ──────
  useEffect(() => {
    const tick = state.currentTick;
    if (!tick || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const tooltip = d3.select(tooltipRef.current);
    const ues = tick.ues ?? [];

    // UE → gNB link lines (behind UE circles)
    const lines = svg.selectAll('.ue-line').data(ues, (d) => d.ue_id);
    lines.enter()
      .append('line')
      .attr('class', 'ue-line')
      .merge(lines)
      .attr('x1', (d) => d.x)
      .attr('y1', (d) => d.y)
      .attr('x2', (d) => GNB[d.connected_cell]?.x ?? 500)
      .attr('y2', (d) => GNB[d.connected_cell]?.y ?? 500)
      .attr('stroke', (d) => CELL_COLORS[d.connected_cell])
      .attr('stroke-opacity', 0.18)
      .attr('stroke-width', 1);
    lines.exit().remove();

    // UE circles
    const circles = svg.selectAll('.ue-circle').data(ues, (d) => d.ue_id);
    circles.enter()
      .append('circle')
      .attr('class', 'ue-circle')
      .attr('r', 6)
      .on('mouseenter', (event, d) => {
        tooltip
          .style('display', 'block')
          .html(
            `<strong>UE ${d.ue_id}</strong><br/>` +
            `Cell: gNB-${d.connected_cell}<br/>` +
            `SINR: ${d.sinr_db} dB<br/>` +
            `Throughput: ${d.throughput_mbps} Mbps` +
            (d.is_handover ? '<br/><span style="color:#f87171">⇄ Handover</span>' : '')
          );
      })
      .on('mousemove', (event) => {
        tooltip
          .style('left', `${event.offsetX + 14}px`)
          .style('top', `${event.offsetY - 10}px`);
      })
      .on('mouseleave', () => tooltip.style('display', 'none'))
      .merge(circles)
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('fill', (d) => CELL_COLORS[d.connected_cell])
      .attr('fill-opacity', 0.88)
      .attr('stroke', (d) => (d.is_handover ? '#ef4444' : '#1f2937'))
      .attr('stroke-width', (d) => (d.is_handover ? 2.5 : 1));
    circles.exit().remove();

  }, [state.currentTick]);

  return (
    <div className="bg-[#111827] rounded-xl border border-gray-800 p-4 relative">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-mono text-gray-400 uppercase tracking-widest">
          Network Map — 1 km × 1 km
        </p>
        <div className="flex gap-3">
          {GNB.map((g) => (
            <span
              key={g.id}
              className="text-xs font-mono flex items-center gap-1"
              style={{ color: CELL_COLORS[g.id] }}
            >
              <span
                className="inline-block w-2 h-2 rounded-sm"
                style={{ background: CELL_COLORS[g.id] }}
              />
              gNB-{g.id}
            </span>
          ))}
          <span className="text-xs font-mono text-red-400 flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full border border-red-400" />
            Handover
          </span>
        </div>
      </div>

      <div className="relative">
        <svg
          ref={svgRef}
          viewBox="0 0 1000 1000"
          className="w-full max-h-[520px]"
          style={{ background: '#060d1a', borderRadius: '8px' }}
        />
        <div
          ref={tooltipRef}
          style={{
            display: 'none',
            position: 'absolute',
            pointerEvents: 'none',
            background: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '6px',
            padding: '8px 12px',
            fontSize: '11px',
            color: '#e5e7eb',
            lineHeight: '1.7',
            zIndex: 10,
            minWidth: '140px',
          }}
        />
      </div>
    </div>
  );
}