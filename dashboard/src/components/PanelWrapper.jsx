import { useState } from 'react';

export default function PanelWrapper({ title, icon, description, hint, children, modal }) {
  const [showModal, setShowModal] = useState(false);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

      {/* Panel header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {/* Title row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {icon && (
              <span style={{ fontSize: '14px', color: 'var(--accent)' }}>{icon}</span>
            )}
            <h2 style={{
              margin: 0,
              fontSize: '11px',
              fontWeight: 700,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
              fontFamily: 'monospace',
            }}>
              {title}
            </h2>
          </div>
          {/* Description */}
          {description && (
            <p style={{
              margin: 0,
              fontSize: '12px',
              color: 'var(--text-secondary)',
              lineHeight: 1.5,
              maxWidth: '560px',
            }}>
              {description}
            </p>
          )}
          {/* Hint to click ⓘ */}
          {hint && (
            <button
              onClick={() => setShowModal(true)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                marginTop: '2px',
                background: 'none',
                border: 'none',
                padding: 0,
                cursor: 'pointer',
                fontSize: '11px',
                color: 'var(--text-muted)',
                fontFamily: 'monospace',
                transition: 'color 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--accent)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <span style={{ fontSize: '12px' }}>ⓘ</span>
              <span>{hint}</span>
            </button>
          )}
        </div>

        {/* Info button on the right */}
        {modal && (
          <button
            onClick={() => setShowModal(true)}
            title="Learn about this panel"
            style={{
              flexShrink: 0,
              width: '28px',
              height: '28px',
              borderRadius: '8px',
              border: '1px solid var(--border-accent)',
              background: 'var(--bg-card)',
              color: 'var(--text-muted)',
              fontSize: '13px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border-accent)'; }}
          >
            ⓘ
          </button>
        )}
      </div>

      {/* Thin accent divider */}
      <div style={{
        height: '1px',
        background: `linear-gradient(to right, var(--accent-glow), transparent)`,
      }} />

      {/* Panel content */}
      <div>
        {children}
      </div>

      {/* Modal */}
      {showModal && modal && (
        <div
          className="modal-backdrop"
          onClick={(e) => { if (e.target === e.currentTarget) setShowModal(false); }}
        >
          <div className="modal-box">
            {/* Modal header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {icon && <span style={{ fontSize: '16px', color: 'var(--accent)' }}>{icon}</span>}
                <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {title}
                </h3>
              </div>
              <button
                onClick={() => setShowModal(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: '18px',
                  cursor: 'pointer',
                  lineHeight: 1,
                  padding: '0 4px',
                }}
              >
                ×
              </button>
            </div>

            {/* Divider */}
            <div style={{ height: '1px', background: 'var(--border)', marginBottom: '16px' }} />

            {/* Modal content */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {modal}
            </div>

            {/* Footer */}
            <div style={{ marginTop: '20px', paddingTop: '12px', borderTop: '1px solid var(--border)' }}>
              <button
                onClick={() => setShowModal(false)}
                style={{
                  padding: '7px 18px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-accent)',
                  background: 'var(--bg-card)',
                  color: 'var(--text-secondary)',
                  fontSize: '12px',
                  cursor: 'pointer',
                  fontFamily: 'monospace',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}