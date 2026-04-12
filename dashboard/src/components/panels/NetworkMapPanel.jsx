import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useSim } from '../../context/SimContext';
import { CELL_COLORS } from '../../constants';

const GNB = [
  { id: 0, x: 250, y: 250 },
  { id: 1, x: 750, y: 250 },
  { id: 2, x: 500, y: 750 },
];

export default function NetworkMapPanel() {
  const { state } = useSim();
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    const tick = state.currentTick;
    if (!tick || !svgRef.current) return;
    const svg = d3.select(svgRef.current);

    const lines = svg.selectAll('.ue-line').data(tick.ues ?? [], (d) => d.ue_id);
    lines
      .enter()
      .append('line')
      .attr('class', 'ue-line')
      .merge(lines)
      .attr('x1', (d) => d.x)
      .attr('y1', (d) => d.y)
      .attr('x2', (d) => GNB[d.connected_cell]?.x ?? 500)
      .attr('y2', (d) => GNB[d.connected_cell]?.y ?? 500)
      .attr('stroke', (d) => CELL_COLORS[d.connected_cell])
      .attr('stroke-opacity', 0.15)
      .attr('stroke-width', 1);
    lines.exit().remove();

    const tooltip = d3.select(tooltipRef.current);
    const circles = svg.selectAll('.ue-circle').data(tick.ues ?? [], (d) => d.ue_id);
    circles
      .enter()
      .append('circle')
      .attr('class', 'ue-circle')
      .attr('r', 7)
      .merge(circles)
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('fill', (d) => CELL_COLORS[d.connected_cell])
      .attr('fill-opacity', 0.85)
      .attr('stroke', (d) => (d.is_handover ? '#ef4444' : 'none'))
      .attr('stroke-width', 2)
      .on('mouseenter', (event, d) => {
        tooltip
          .style('display', 'block')
          .html(
            `UE ${d.ue_id}<br/>Cell: ${d.connected_cell}<br/>SINR: ${d.sinr_db} dB<br/>${d.throughput_mbps} Mbps`,
          );
      })
      .on('mousemove', (event) => {
        tooltip.style('left', `${event.offsetX + 12}px`).style('top', `${event.offsetY - 10}px`);
      })
      .on('mouseleave', () => tooltip.style('display', 'none'));
    circles.exit().remove();

    const gnbs = svg.selectAll('.gnb').data(GNB, (d) => d.id);
    gnbs
      .enter()
      .append('polygon')
      .attr('class', 'gnb')
      .merge(gnbs)
      .attr('points', (d) => `${d.x},${d.y - 18} ${d.x - 14},${d.y + 10} ${d.x + 14},${d.y + 10}`)
      .attr('fill', (d) => CELL_COLORS[d.id])
      .attr('fill-opacity', 0.9);

    const labels = svg.selectAll('.gnb-label').data(GNB, (d) => d.id);
    labels
      .enter()
      .append('text')
      .attr('class', 'gnb-label')
      .merge(labels)
      .attr('x', (d) => d.x)
      .attr('y', (d) => d.y + 26)
      .attr('text-anchor', 'middle')
      .attr('fill', (d) => CELL_COLORS[d.id])
      .attr('font-size', 11)
      .attr('font-family', 'monospace')
      .text((d) => `gNB-${d.id}`);
  }, [state.currentTick]);

  return (
    <div className="bg-[#111827] rounded-xl border border-gray-800 p-4 relative">
      <p className="text-xs font-mono text-gray-400 mb-3 uppercase tracking-widest">
        Network Map — 1km × 1km
      </p>
      <div className="relative">
        <svg
          ref={svgRef}
          viewBox="0 0 1000 1000"
          className="w-full max-h-[520px]"
          style={{ background: '#0a0f1e', borderRadius: '8px' }}
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
            padding: '8px 10px',
            fontSize: '11px',
            color: '#e5e7eb',
            lineHeight: '1.6',
            zIndex: 10,
          }}
        />
      </div>
    </div>
  );
}
