import { Shield, Activity, Eye } from 'lucide-react';
import './SystemHealth.css';

const SYSTEMS = [
  { name: 'DLP', status: 'Active', desc: 'Scanning for PII', icon: <Shield size={16} />, color: 'var(--accent-green)' },
  { name: 'IDS', status: 'Active', desc: 'Rate monitoring', icon: <Activity size={16} />, color: 'var(--accent-green)' },
  { name: 'UEBA', status: 'Active', desc: 'Behavioral baseline', icon: <Eye size={16} />, color: 'var(--accent-green)' },
];

export default function SystemHealth() {
  return (
    <div className="system-health">
      <h2>System Health</h2>
      <p className="text-muted" style={{ fontSize: '13px', marginBottom: '20px' }}>
        DLP / IDS / UEBA monitoring status
      </p>
      <div className="health-grid">
        {SYSTEMS.map(sys => (
          <div key={sys.name} className="health-card">
            <div className="health-card-header">
              <div className="health-icon">{sys.icon}</div>
              <span className="health-name font-mono">{sys.name}</span>
              <span className="health-status" style={{ color: sys.color }}>
                <span className="health-dot" style={{ background: sys.color }} />
                {sys.status}
              </span>
            </div>
            <p className="health-desc">— {sys.desc}</p>
          </div>
        ))}
      </div>

      <div className="health-meta">
        <div className="health-meta-item">
          <span className="health-meta-label">Last Check</span>
          <span className="health-meta-value font-mono">{new Date().toLocaleTimeString()}</span>
        </div>
        <div className="health-meta-item">
          <span className="health-meta-label">Uptime</span>
          <span className="health-meta-value font-mono">99.9%</span>
        </div>
        <div className="health-meta-item">
          <span className="health-meta-label">API Latency</span>
          <span className="health-meta-value font-mono">42ms</span>
        </div>
      </div>
    </div>
  );
}
