import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useSim } from '../../context/SimContext';
import { CELL_COLORS } from '../../constants';
import PanelWrapper from '../PanelWrapper';

const GNB = [
  { id: 0, x: 200, y: 500 },
  { id: 1, x: 500, y: 200 },
  { id: 2, x: 800, y: 700 },
];

const COVERAGE_RADIUS = 280;

const PROFILE_COLORS = {
  Video:  '#3b82f6',
  Gaming: '#22c55e',
  IoT:    '#eab308',
  VoIP:   '#a855f7',
};

const PROFILE_LEGEND = [
  { label: 'Video',  color: '#3b82f6' },
  { label: 'Gaming', color: '#22c55e' },
  { label: 'IoT',    color: '#eab308' },
  { label: 'VoIP',   color: '#a855f7' },
];

export default function NetworkMapPanel() {
  const { state } = useSim();
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const staticRenderedRef = useRef(false);

  useEffect(() => {
    if (!svgRef.current || staticRenderedRef.current) return;
    staticRenderedRef.current = true;

    const svg = d3.select(svgRef.current);

    const gridGroup = svg.append('g').attr('class', 'grid-layer');
    for (let i = 0; i <= 10; i++) {
      gridGroup.append('line')
        .attr('x1', i * 100).attr('y1', 0)
        .attr('x2', i * 100).attr('y2', 1000)
        .attr('stroke', 'rgba(255,255,255,0.03)')
        .attr('stroke-width', 1);
      gridGroup.append('line')
        .attr('x1', 0).attr('y1', i * 100)
        .attr('x2', 1000).attr('y2', i * 100)
        .attr('stroke', 'rgba(255,255,255,0.03)')
        .attr('stroke-width', 1);
    }

    svg.selectAll('.coverage-ring-outer')
      .data(GNB).enter().append('circle')
      .attr('class', 'coverage-ring-outer')
      .attr('cx', d => d.x).attr('cy', d => d.y)
      .attr('r', COVERAGE_RADIUS)
      .attr('fill', d => CELL_COLORS[d.id])
      .attr('fill-opacity', 0.04)
      .attr('stroke', d => CELL_COLORS[d.id])
      .attr('stroke-opacity', 0.12)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '8 5');

    svg.selectAll('.coverage-ring-inner')
      .data(GNB).enter().append('circle')
      .attr('class', 'coverage-ring-inner')
      .attr('cx', d => d.x).attr('cy', d => d.y)
      .attr('r', COVERAGE_RADIUS * 0.55)
      .attr('fill', d => CELL_COLORS[d.id])
      .attr('fill-opacity', 0.05)
      .attr('stroke', 'none');

    svg.selectAll('.gnb-tower')
      .data(GNB).enter().append('polygon')
      .attr('class', 'gnb-tower')
      .attr('points', d =>
        `${d.x},${d.y - 22} ${d.x - 16},${d.y + 13} ${d.x + 16},${d.y + 13}`
      )
      .attr('fill', d => CELL_COLORS[d.id])
      .attr('fill-opacity', 0.92)
      .attr('stroke', d => CELL_COLORS[d.id])
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.6)
      .style('filter', d => `drop-shadow(0 0 8px ${CELL_COLORS[d.id]}88)`);

    svg.selectAll('.gnb-base')
      .data(GNB).enter().append('circle')
      .attr('class', 'gnb-base')
      .attr('cx', d => d.x).attr('cy', d => d.y + 13)
      .attr('r', 18)
      .attr('fill', 'none')
      .attr('stroke', d => CELL_COLORS[d.id])
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 1);

    svg.selectAll('.gnb-label')
      .data(GNB).enter().append('text')
      .attr('class', 'gnb-label')
      .attr('x', d => d.x)
      .attr('y', d => d.y + 38)
      .attr('text-anchor', 'middle')
      .attr('fill', d => CELL_COLORS[d.id])
      .attr('font-size', 11)
      .attr('font-family', 'monospace')
      .attr('font-weight', 'bold')
      .attr('letter-spacing', '0.05em')
      .text(d => `gNB-${d.id}`);

    svg.selectAll('.gnb-sublabel')
      .data(GNB).enter().append('text')
      .attr('class', 'gnb-sublabel')
      .attr('x', d => d.x)
      .attr('y', d => d.y + 52)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.25)')
      .attr('font-size', 9)
      .attr('font-family', 'monospace')
      .text('3.5 GHz');

  }, []);

  useEffect(() => {
    const tick = state.currentTick;
    if (!tick || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const tooltip = d3.select(tooltipRef.current);
    const ues = tick.ues ?? [];

    const lines = svg.selectAll('.ue-line').data(ues, d => d.ue_id);
    lines.enter().append('line').attr('class', 'ue-line')
      .merge(lines)
      .attr('x1', d => d.x).attr('y1', d => d.y)
      .attr('x2', d => GNB[d.connected_cell]?.x ?? 500)
      .attr('y2', d => GNB[d.connected_cell]?.y ?? 500)
      .attr('stroke', d => CELL_COLORS[d.connected_cell])
      .attr('stroke-opacity', 0.2)
      .attr('stroke-width', 0.8);
    lines.exit().remove();

    const glows = svg.selectAll('.ue-glow').data(ues, d => d.ue_id);
    glows.enter().append('circle').attr('class', 'ue-glow').attr('r', 10)
      .merge(glows)
      .attr('cx', d => d.x).attr('cy', d => d.y)
      .attr('fill', 'none')
      .attr('stroke', d => d.is_handover ? '#ef4444' : 'transparent')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.8);
    glows.exit().remove();

    const circles = svg.selectAll('.ue-circle').data(ues, d => d.ue_id);
    circles.enter().append('circle').attr('class', 'ue-circle').attr('r', 5.5)
      .on('mouseenter', (event, d) => {
        const pc = PROFILE_COLORS[d.traffic_profile] ?? '#94a3b8';
        tooltip.style('display', 'block').html(
          `<div style="font-weight:700;margin-bottom:4px;color:#f9fafb">UE ${d.ue_id}</div>` +
          `<div style="color:${pc};margin-bottom:2px">● ${d.traffic_profile ?? 'Unknown'} (5QI-${d.qos_class ?? '?'})</div>` +
          `<div>Cell: <span style="color:${CELL_COLORS[d.connected_cell]}">gNB-${d.connected_cell}</span></div>` +
          `<div>SINR: <span style="color:#00d4ff">${d.sinr_db} dB</span></div>` +
          `<div>Throughput: <span style="color:#10b981">${d.throughput_mbps} Mbps</span></div>` +
          (d.is_handover ? `<div style="color:#f87171;margin-top:3px">⇄ Handover active</div>` : '')
        );
      })
      .on('mousemove', (event) => {
        tooltip.style('left', `${event.offsetX + 14}px`).style('top', `${event.offsetY - 10}px`);
      })
      .on('mouseleave', () => tooltip.style('display', 'none'))
      .merge(circles)
      .attr('cx', d => d.x).attr('cy', d => d.y)
      .attr('fill', d => PROFILE_COLORS[d.traffic_profile] ?? '#94a3b8')
      .attr('fill-opacity', 0.9)
      .attr('stroke', d => d.is_handover ? '#ef4444' : 'rgba(255,255,255,0.15)')
      .attr('stroke-width', d => d.is_handover ? 2 : 0.8)
      .style('filter', d => `drop-shadow(0 0 4px ${PROFILE_COLORS[d.traffic_profile] ?? '#94a3b8'}99)`);
    circles.exit().remove();

  }, [state.currentTick]);

  return (
    <PanelWrapper
      title="Network Map"
      icon="⬡"
      description="Live D3.js visualization of the 1km × 1km grid. Each dot is a mobile UE moving with Random Waypoint mobility. Lines connect UEs to their serving gNB. Colour = 5QI traffic profile."
      hint="Hover a UE dot to see SINR and throughput"
      modal={
        <>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            This map renders exact UE and gNB positions from the SimPy simulation, updated every tick via <span style={{ color: 'var(--accent)', fontWeight: 600 }}>WebSocket</span>. The 3 triangles are gNB base stations; the coloured dots are mobile UEs.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>5QI colour coding (3GPP TS 23.501):</span> 🔵 Video (5QI-2, 10–20 Mbps) · 🟢 Gaming (5QI-3) · 🟡 IoT (5QI-5) · 🟣 VoIP (5QI-1)
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: '#ef4444', fontWeight: 600 }}>Red ring</span> = active handover. Triggers when a neighbour gNB offers SINR {'>'} 3 dB higher (hysteresis-based, standard in LTE/NR).
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            D3.js · 20 UEs · Random Waypoint · max 3 m/s · α=3.5 path loss
          </p>
        </>
      }
    >
      {/* Legend — wraps cleanly on mobile */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: '10px',
        marginBottom: '12px',
        rowGap: '8px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          {GNB.map(g => (
            <span key={g.id} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', fontFamily: 'monospace', color: CELL_COLORS[g.id] }}>
              <span style={{ width: 8, height: 8, background: CELL_COLORS[g.id], clipPath: 'polygon(50% 0%,0% 100%,100% 100%)', display: 'inline-block' }} />
              gNB-{g.id}
            </span>
          ))}
        </div>
        <div style={{ width: 1, height: 14, background: 'var(--border)', flexShrink: 0 }} />
        {PROFILE_LEGEND.map(p => (
          <span key={p.label} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', fontFamily: 'monospace', color: p.color }}>
            <span style={{ width: 8, height: 8, background: p.color, borderRadius: '50%', display: 'inline-block' }} />
            {p.label}
          </span>
        ))}
        <div style={{ width: 1, height: 14, background: 'var(--border)', flexShrink: 0 }} />
        <span style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', fontFamily: 'monospace', color: '#ef4444' }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', border: '1.5px solid #ef4444', display: 'inline-block' }} />
          Handover
        </span>
      </div>

      {/* Map container */}
      <div
        style={{
          position: 'relative',
          borderRadius: '12px',
          overflow: 'hidden',
          border: '1px solid var(--border)',
          background: 'var(--map-bg, #060d1a)',
        }}
      >
        <div
          style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: 3,
            background: 'linear-gradient(90deg, #00d4ff33, #8b5cf633, #10b98133)',
            zIndex: 2,
          }}
        />

        <svg
          ref={svgRef}
          viewBox="0 0 1000 1000"
          style={{
            width: '100%',
            maxHeight: '55vw',
            minHeight: '260px',
            display: 'block',
          }}
        />

        <div
          ref={tooltipRef}
          style={{
            display: 'none',
            position: 'absolute',
            pointerEvents: 'none',
            background: 'rgba(10,15,30,0.95)',
            border: '1px solid rgba(0,212,255,0.3)',
            borderRadius: 8,
            padding: '10px 14px',
            fontSize: 11,
            color: '#e5e7eb',
            lineHeight: 1.8,
            zIndex: 10,
            minWidth: 150,
            backdropFilter: 'blur(8px)',
            boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
          }}
        />

        <div
          style={{
            position: 'absolute', bottom: 8, right: 10,
            fontSize: 10, fontFamily: 'monospace',
            color: 'rgba(255,255,255,0.2)',
            zIndex: 2,
          }}
        >
          TICK #{state.currentTick?.tick ?? '—'} · 1 km × 1 km
        </div>
      </div>
    </PanelWrapper>
  );
}