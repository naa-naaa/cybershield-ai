import { Shield, Bell, LogOut, Zap, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { logout } from '../../lib/store';
import './ChatHeader.css';

interface ChatHeaderProps {
  shieldOn: boolean;
  shieldOffline: boolean;
  onToggleShield: () => void;
  strikeCount: number;
  notificationCount: number;
  onNotificationClick: () => void;
  currentUser: string;
}

export default function ChatHeader({
  shieldOn,
  shieldOffline,
  onToggleShield,
  strikeCount,
  notificationCount,
  onNotificationClick,
  currentUser,
}: ChatHeaderProps) {
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="chat-header">
      <div className="chat-header-left">
        <div className="chat-header-logo">
          <Shield size={22} />
          <span className="chat-header-name font-heading">SafeChat</span>
        </div>
        <span className="chat-header-user">@{currentUser}</span>
      </div>

      <div className="chat-header-center">
        <div className="shield-toggle-wrap">
          <label className="shield-toggle" htmlFor="shield-toggle">
            <input
              id="shield-toggle"
              type="checkbox"
              checked={shieldOn}
              onChange={onToggleShield}
            />
            <span className="shield-toggle-slider" />
          </label>
          <span className="shield-toggle-label">
            CyberShield {shieldOn ? 'ON' : 'OFF'}
          </span>
        </div>
        
        {shieldOn && !shieldOffline && (
          <span className="shield-active-pill animate-fade-in">
            <span className="shield-dot green" />
            CyberShield Active
          </span>
        )}
        
        {shieldOffline && (
          <span className="shield-offline-pill animate-fade-in">
            <span className="shield-dot gray" />
            CyberShield Offline
          </span>
        )}
      </div>

      <div className="chat-header-right">
        {strikeCount >= 3 ? (
          <span className="strike-badge strike-restricted animate-flash">
            <AlertCircle size={14} />
            🔴 Account Restricted
          </span>
        ) : (
          <span className={`strike-badge ${strikeCount > 0 ? 'strike-warning' : ''}`}>
            <Zap size={14} />
            {strikeCount} strikes
          </span>
        )}

        <button
          className="header-icon-btn notification-btn"
          onClick={onNotificationClick}
          aria-label="Notifications"
        >
          <Bell size={20} />
          {notificationCount > 0 && (
            <span className="notification-badge">{notificationCount}</span>
          )}
        </button>

        <button
          className="header-icon-btn logout-btn"
          onClick={handleLogout}
          aria-label="Log out"
        >
          <LogOut size={18} />
        </button>
      </div>
    </header>
  );
}
