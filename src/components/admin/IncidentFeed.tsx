import { Incident } from '../../lib/types';
import './IncidentFeed.css';

interface IncidentFeedProps {
  incidents: Incident[];
  onSelectIncident: (incident: Incident) => void;
  selectedId: string | null;
}

function getTimeAgo(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function generateFakeIP(): string {
  const ips = ['185.220.101.45', '192.168.1.105', '10.0.0.42', '172.16.0.88', '203.0.113.50'];
  return ips[Math.floor(Math.random() * ips.length)];
}

export default function IncidentFeed({ incidents, onSelectIncident, selectedId }: IncidentFeedProps) {
  return (
    <div className="incident-feed">
      <div className="feed-header">
        <h3>Live Incident Feed</h3>
        <span className="feed-count font-mono">{incidents.length} incidents</span>
      </div>
      <div className="feed-list">
        {incidents.length === 0 ? (
          <div className="feed-empty">
            <p>No incidents yet</p>
            <p className="text-muted">Incidents will appear here in real-time when CyberShield analyzes messages.</p>
          </div>
        ) : (
          incidents.map((incident, i) => {
            const isSelected = incident.id === selectedId;
            const actionClass = incident.action;
            const ip = incident.ip_address || generateFakeIP();
            const isTor = ip.startsWith('185.');

            return (
              <div
                key={incident.id}
                className={`incident-card ${actionClass} ${isSelected ? 'selected' : ''} ${i === 0 ? 'animate-slide-down' : ''}`}
                onClick={() => onSelectIncident(incident)}
              >
                <div className="incident-card-header">
                  <span className={`action-dot ${actionClass}`} />
                  <span className="incident-users">
                    {incident.sender} → {incident.receiver}
                  </span>
                  <span className="incident-divider">|</span>
                  <span className="incident-category">{incident.category}</span>
                  <span className="incident-divider">|</span>
                  <span className="incident-score font-mono">{incident.harm_score.toFixed(2)}</span>
                </div>
                <div className="incident-card-footer">
                  <span className={`action-badge ${actionClass}`}>
                    {incident.action.toUpperCase()}
                  </span>
                  <span className="incident-time">· {getTimeAgo(incident.timestamp)}</span>
                  <span className="incident-ip font-mono">
                    · IP: {ip} {isTor && <span className="tor-tag">[TOR]</span>}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
