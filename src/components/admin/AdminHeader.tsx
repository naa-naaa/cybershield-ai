import { Shield } from 'lucide-react';
import { AlertLevel } from '../../lib/types';
import './AdminHeader.css';

interface AdminHeaderProps {
  alertLevel: AlertLevel;
  activeAlerts: number;
  blockedToday: number;
  protectedUsers: number;
}

const LEVEL_CONFIG: Record<AlertLevel, { emoji: string; label: string; color: string }> = {
  GREEN: { emoji: '🟢', label: 'GREEN', color: 'var(--accent-green)' },
  YELLOW: { emoji: '🟡', label: 'YELLOW', color: 'var(--accent-amber)' },
  ORANGE: { emoji: '🟠', label: 'ORANGE', color: '#FF8C00' },
  RED: { emoji: '🔴', label: 'RED', color: 'var(--accent-red)' },
};

const LEVEL_DESC: Record<AlertLevel, string> = {
  GREEN: 'No active threats',
  YELLOW: '2+ alerts in last 10 min',
  ORANGE: 'Active harassment pattern',
  RED: 'Crisis level — immediate action required',
};

export default function AdminHeader({
  alertLevel,
  activeAlerts,
  blockedToday,
  protectedUsers,
}: AdminHeaderProps) {
  const level = LEVEL_CONFIG[alertLevel];

  return (
    <header className="admin-header">
      <div className="admin-header-left">
        <div className="admin-logo">
          <Shield size={24} />
          <span className="admin-wordmark font-heading">🛡 CyberShield Admin</span>
        </div>
      </div>

      <div className="admin-header-center">
        <div
          className="rakshak-badge"
          style={{ borderColor: level.color }}
        >
          <span className="rakshak-emoji">{level.emoji}</span>
          <div className="rakshak-info">
            <span className="rakshak-label" style={{ color: level.color }}>
              Rakshak Alert: {level.label}
            </span>
            <span className="rakshak-desc">{LEVEL_DESC[alertLevel]}</span>
          </div>
        </div>
      </div>

      <div className="admin-header-right">
        <div className="admin-stat">
          <span className="stat-value font-mono text-amber">{activeAlerts}</span>
          <span className="stat-label">Active Alerts</span>
        </div>
        <div className="admin-stat-divider" />
        <div className="admin-stat">
          <span className="stat-value font-mono text-red">{blockedToday}</span>
          <span className="stat-label">Blocked Today</span>
        </div>
        <div className="admin-stat-divider" />
        <div className="admin-stat">
          <span className="stat-value font-mono text-cyan">{protectedUsers}</span>
          <span className="stat-label">Protected Users</span>
        </div>
      </div>
    </header>
  );
}
