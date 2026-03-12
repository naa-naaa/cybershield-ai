import { AlertTriangle, X } from 'lucide-react';
import { Incident } from '../../lib/types';
import './BlockToast.css';

interface BlockToastProps {
  incident: Incident;
  onDismiss: () => void;
  onViewCase: () => void;
}

export default function BlockToast({ incident, onDismiss, onViewCase }: BlockToastProps) {
  return (
    <div className="block-toast animate-slide-in-right">
      <button className="toast-close" onClick={onDismiss}>
        <X size={14} />
      </button>
      <div className="toast-header">
        <AlertTriangle size={18} />
        <span className="toast-title">🚨 USER BLOCKED</span>
      </div>
      <div className="toast-body">
        <p className="toast-users">{incident.sender} → {incident.receiver}</p>
        <p className="toast-detail">
          {incident.category} · Score: <span className="font-mono">{incident.harm_score.toFixed(2)}</span>
        </p>
      </div>
      <div className="toast-actions">
        <button className="toast-btn toast-btn-view" onClick={onViewCase}>View Case</button>
        <button className="toast-btn toast-btn-dismiss" onClick={onDismiss}>Dismiss</button>
      </div>
    </div>
  );
}
