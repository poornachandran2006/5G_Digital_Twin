import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useSim } from '../../context/SimContext';
import { CELL_COLORS } from '../../constants';
import { useState } from 'react';
import InfoModal from '../InfoModal';
import PanelWrapper from '../PanelWrapper';
// gNB positions are fixed — match simulation/engine.py _GNB_POSITIONS
// engine uses (200,500), (500,200), (800,700) on a 1000x1000 grid
const GNB = [
  { id: 0, x: 200, y: 500 },
  { id: 1, x: 500, y: 200 },
  { id: 2, x: 800, y: 700 },
];

// Coverage radius visual guide (not physics — purely decorative)
const COVERAGE_RADIUS = 280;

// Traffic profile colors — 3GPP 5QI aligned
const PROFILE_COLORS = {
  Video: '#3b82f6',  // blue   — 5QI-2
  Gaming: '#22c55e',  // green  — 5QI-3
  IoT: '#eab308',  // yellow — 5QI-5
  VoIP: '#a855f7',  // purple — 5QI-1
};

const PROFILE_LEGEND = [
  { label: 'Video', color: '#3b82f6' },
  { label: 'Gaming', color: '#22c55e' },
  { label: 'IoT', color: '#eab308' },
  { label: 'VoIP', color: '#a855f7' },
];

export default function NetworkMapPanel() {
  const { state } = useSim();
  const [showInfo, setShowInfo] = useState(false);
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
        const profileColor = PROFILE_COLORS[d.traffic_profile] ?? '#94a3b8';
        tooltip
          .style('display', 'block')
          .html(
            `<strong>UE ${d.ue_id}</strong><br/>` +
            `<span style="color:${profileColor}">● ${d.traffic_profile ?? 'Unknown'} (5QI-${d.qos_class ?? '?'})</span><br/>` +
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
      .attr('fill', (d) => PROFILE_COLORS[d.traffic_profile] ?? '#94a3b8')
      .attr('fill-opacity', 0.88)
      .attr('stroke', (d) => (d.is_handover ? '#ef4444' : '#1f2937'))
      .attr('stroke-width', (d) => (d.is_handover ? 2.5 : 1));
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
            This map renders the exact UE and gNB positions from the SimPy simulation engine, updated every tick via <span style={{ color: 'var(--accent)', fontWeight: 600 }}>WebSocket</span>.
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Colour coding (5QI profiles per 3GPP TS 23.501):</span><br />
            🔵 Blue = Video (5QI-2, 10–20 Mbps) · 🟢 Green = Gaming (5QI-3) · 🟡 Yellow = IoT (5QI-5) · 🟣 Purple = VoIP (5QI-1)
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Red stroke</span> = active handover event. A handover triggers when another gNB offers SINR more than 3 dB higher (hysteresis-based, standard in LTE/5G).
          </p>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>SINR formula:</span> Received power from serving gNB ÷ (noise −104 dBm + interference from other gNBs). Path loss exponent α = 3.5 (3GPP TR 38.901 urban macro model).
          </p>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: '4px' }}>
            Rendered with D3.js · 20 UEs · Random Waypoint mobility · max 3 m/s
          </p>
        </>
      }
    >
      <div className="bg-[#111827] rounded-xl border border-gray-800 p-4 relative">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <p className="text-xs font-mono text-gray-400 uppercase tracking-widest">
              Network Map — 1 km × 1 km
            </p>
            <button
              onClick={() => setShowInfo(true)}
              className="text-gray-500 hover:text-green-400 transition-colors text-sm"
              title="What is this panel?"
            >
              ⓘ
            </button>
          </div>

          {showInfo && (
            <InfoModal title="Network Map Panel" onClose={() => setShowInfo(false)}>
              <p>
                This is a <span className="text-green-400 font-semibold">live top-down view</span> of
                the simulated 1km × 1km city. Every dot is a mobile phone (UE), every triangle is a
                cell tower (gNB).
              </p>
              <p>
                <span className="text-white font-semibold">UE dot colors</span> show what each user
                is doing — 🔵 Video streaming, 🟢 Gaming, 🟡 IoT sensor, 🟣 VoIP call. These are
                3GPP 5QI traffic profiles from the real 5G standard.
              </p>
              <p>
                <span className="text-white font-semibold">Lines</span> connect each phone to its
                serving tower. A <span className="text-red-400 font-semibold">red ring</span> around
                a dot means that phone is currently switching towers (handover event).
              </p>
              <p>
                <span className="text-white font-semibold">Dashed rings</span> around towers are
                visual coverage guides — not physics. Actual signal strength is computed per-UE using
                the 3GPP path loss formula at 3.5 GHz.
              </p>
              <p className="text-gray-500 text-xs pt-1">
                Phones move using the Random Waypoint mobility model — each picks a random destination
                and walks toward it at up to 3 m/s, then picks a new one.
              </p>
            </InfoModal>
          )}
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
          {/* Traffic profile legend */}
          <div className="flex gap-3 mt-1">
            {PROFILE_LEGEND.map((p) => (
              <span
                key={p.label}
                className="text-xs font-mono flex items-center gap-1"
                style={{ color: p.color }}
              >
                <span
                  className="inline-block w-2 h-2 rounded-full"
                  style={{ background: p.color }}
                />
                {p.label}
              </span>
            ))}
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
    </PanelWrapper>
  );

}