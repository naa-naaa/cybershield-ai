import { AlertTriangle, Edit3, SendHorizontal, X } from 'lucide-react';
import './AlertBanner.css';

interface AlertBannerProps {
  category: string;
  harmScore: number;
  onSendAnyway: () => void;
  onEdit: () => string;
  onCancel: () => void;
}

export default function AlertBanner({
  category,
  harmScore,
  onSendAnyway,
  onEdit,
  onCancel,
}: AlertBannerProps) {
  return (
    <div className="alert-banner animate-slide-down">
      <div className="alert-banner-content">
        <div className="alert-banner-icon">
          <AlertTriangle size={20} />
        </div>
        <div className="alert-banner-info">
          <h4 className="alert-banner-title">⚠ CyberShield Alert</h4>
          <p className="alert-banner-desc">
            <span className="alert-category">{category}</span> detected · 
            <span className="alert-score font-mono"> {Math.round(harmScore * 100)}%</span>
          </p>
        </div>
        <div className="alert-banner-actions">
          <button className="alert-btn alert-btn-edit" onClick={onEdit}>
            <Edit3 size={14} />
            Edit Message
          </button>
          <button className="alert-btn alert-btn-send" onClick={onSendAnyway}>
            <SendHorizontal size={14} />
            Send Anyway
          </button>
          <button className="alert-btn alert-btn-cancel" onClick={onCancel}>
            <X size={14} />
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
