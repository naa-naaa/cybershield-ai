import { Ban, PauseCircle, CheckCircle, FileText, AlertTriangle } from 'lucide-react';
import { Incident } from '../../lib/types';
import { generateEvidenceReport } from '../../lib/store';
import './CaseDetail.css';

interface CaseDetailProps {
  incident: Incident;
}

function ScoreBar({ label, score, color }: { label: string; score: number | null; color: string }) {
  const pct = score != null ? Math.round(score * 100) : 0;
  return (
    <div className="score-bar-row">
      <span className="score-bar-label">{label}</span>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="score-bar-value font-mono">{score?.toFixed(2) ?? 'N/A'}</span>
    </div>
  );
}

export default function CaseDetail({ incident }: CaseDetailProps) {
  const handleExport = () => {
    const report = generateEvidenceReport(incident);
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CS-${incident.id.slice(0, 8).toUpperCase()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReport = () => {
    handleExport();
    window.open('https://cybercrime.gov.in', '_blank');
  };

  return (
    <div className="case-detail">
      <div className="case-detail-header">
        <h3>Case Detail</h3>
        <span className={`case-action-badge ${incident.action}`}>
          {incident.action.toUpperCase()}
        </span>
      </div>

      <div className="case-section">
        <h4>Agent Analysis</h4>
        <div className="score-bars">
          <ScoreBar label="T1 Toxicity" score={incident.agent_t1} color="var(--accent-red)" />
          <ScoreBar label="T2 Context" score={incident.agent_t2} color="var(--accent-amber)" />
          <ScoreBar label="T3 Emotion" score={incident.agent_t3} color="var(--accent-cyan)" />
          <div className="score-bar-divider" />
          <ScoreBar
            label="FINAL"
            score={incident.harm_score}
            color={incident.action === 'block' ? 'var(--accent-red)' : incident.action === 'alert' ? 'var(--accent-amber)' : 'var(--accent-green)'}
          />
        </div>
      </div>

      <div className="case-section">
        <h4>Message Content</h4>
        <div className="case-message-box">
          <p>{incident.content}</p>
        </div>
        <div className="case-meta">
          <span><strong>Sender:</strong> {incident.sender}</span>
          <span><strong>Receiver:</strong> {incident.receiver}</span>
          <span><strong>Category:</strong> {incident.category}</span>
          <span><strong>Time:</strong> {new Date(incident.timestamp).toLocaleString()}</span>
        </div>
      </div>

      <div className="case-section">
        <h4>IP Metadata</h4>
        <div className="ip-card">
          <div className="ip-row"><span className="ip-label">Address</span><span className="ip-value font-mono">{incident.ip_address || '185.220.101.45'}</span></div>
          <div className="ip-row"><span className="ip-label">Network</span><span className="ip-value">TOR Exit Node</span></div>
          <div className="ip-row"><span className="ip-label">Country</span><span className="ip-value">India 🇮🇳</span></div>
          <div className="ip-row"><span className="ip-label">Risk</span><span className="ip-value text-red">High</span></div>
        </div>
      </div>

      <div className="case-actions">
        <button className="case-btn case-btn-ban">
          <Ban size={14} />
          Permanent Ban
        </button>
        <button className="case-btn case-btn-suspend">
          <PauseCircle size={14} />
          Suspend 24hr
        </button>
        <button className="case-btn case-btn-override">
          <CheckCircle size={14} />
          Override - False Positive
        </button>
        <button className="case-btn case-btn-export" onClick={handleExport}>
          <FileText size={14} />
          Export Evidence PDF
        </button>
        <button className="case-btn case-btn-report" onClick={handleReport}>
          <AlertTriangle size={14} />
          Report to Cybercrime Cell
        </button>
      </div>
    </div>
  );
}
