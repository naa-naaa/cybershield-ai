import { ShieldOff } from 'lucide-react';
import './BlockOverlay.css';

interface BlockOverlayProps {
  strikeNum: number;
  category: string;
  cooldown: number;
  onDismiss: () => void;
}

export default function BlockOverlay({
  strikeNum,
  category,
  cooldown,
  onDismiss,
}: BlockOverlayProps) {
  const minutes = String(Math.floor(cooldown / 60)).padStart(2, '0');
  const seconds = String(cooldown % 60).padStart(2, '0');

  return (
    <div className="block-overlay animate-fade-in">
      <div className="block-card animate-scale-in">
        <div className="block-icon">
          <ShieldOff size={48} />
        </div>
        
        <h2 className="block-title">🚫 Message Blocked</h2>
        <p className="block-subtitle">CyberShield has blocked this message.</p>
        
        <div className="block-strike font-mono">
          This is strike <span className="strike-num">{strikeNum}</span> of 3.
        </div>
        
        <div className="block-reason">
          Reason: <span className="reason-category">{category}</span>
        </div>

        <div className="block-divider" />

        <blockquote className="block-quote">
          "Words have power. Take a moment."
        </blockquote>
        
        <p className="block-stat">
          In India, 1 in 4 students experience cyberbullying. Your message was stopped before it could cause harm.
        </p>

        <div className="block-divider" />

        <div className="block-countdown">
          <span className="countdown-label">Cooldown:</span>
          <span className="countdown-timer font-mono">{minutes}:{seconds}</span>
        </div>

        <button
          className="block-dismiss-btn"
          onClick={onDismiss}
          disabled={cooldown > 0}
        >
          I understand. I will be more mindful.
        </button>
      </div>
    </div>
  );
}
