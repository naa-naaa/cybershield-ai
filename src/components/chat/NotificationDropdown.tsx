import { AlertTriangle, ShieldOff, Check } from 'lucide-react';
import { Notification } from '../../lib/types';
import './NotificationDropdown.css';

interface NotificationDropdownProps {
  notifications: Notification[];
  onClose: () => void;
  onMarkRead: (id: string) => void;
}

export default function NotificationDropdown({
  notifications,
  onClose,
  onMarkRead,
}: NotificationDropdownProps) {
  return (
    <>
      <div className="notif-backdrop" onClick={onClose} />
      <div className="notif-dropdown animate-scale-in">
        <div className="notif-header">
          <h4>Notifications</h4>
          <span className="notif-count">{notifications.filter(n => !n.read).length} unread</span>
        </div>
        <div className="notif-list">
          {notifications.length === 0 ? (
            <div className="notif-empty">
              <p>No notifications</p>
            </div>
          ) : (
            notifications.map(notif => (
              <div
                key={notif.id}
                className={`notif-item ${notif.read ? 'read' : 'unread'}`}
                onClick={() => onMarkRead(notif.id)}
              >
                <div className={`notif-icon ${notif.type}`}>
                  {notif.type === 'blocked' ? (
                    <ShieldOff size={14} />
                  ) : (
                    <AlertTriangle size={14} />
                  )}
                </div>
                <div className="notif-content">
                  <p className="notif-message">{notif.message}</p>
                  <span className="notif-time font-mono">
                    {new Date(notif.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span>
                </div>
                {!notif.read && (
                  <div className="notif-unread-dot" />
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
}
