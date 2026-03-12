import { useState, useEffect, useRef, useCallback } from 'react';
import { Incident, AdminSection, AlertLevel } from '../lib/types';
import { getIncidents, subscribeToIncidents } from '../lib/store';
import AdminHeader from '../components/admin/AdminHeader';
import AdminSidebar from '../components/admin/AdminSidebar';
import IncidentFeed from '../components/admin/IncidentFeed';
import CaseDetail from '../components/admin/CaseDetail';
import UserProfiles from '../components/admin/UserProfiles';
import SystemHealth from '../components/admin/SystemHealth';
import BlockToast from '../components/admin/BlockToast';
import './AdminPage.css';

// Chime sound (base64 encoded short beep)
const CHIME_AUDIO = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACA';

export default function AdminPage() {
  const [incidents, setIncidents] = useState<Incident[]>(getIncidents());
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [activeSection, setActiveSection] = useState<AdminSection>('feed');
  const [blockToast, setBlockToast] = useState<Incident | null>(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Calculate alert level
  const calculateAlertLevel = useCallback((): AlertLevel => {
    const now = Date.now();
    const tenMinAgo = now - 10 * 60 * 1000;
    const recentIncidents = incidents.filter(i => new Date(i.timestamp).getTime() > tenMinAgo);
    const blocks = recentIncidents.filter(i => i.action === 'block');
    const alerts = recentIncidents.filter(i => i.action === 'alert');

    if (blocks.length >= 2) return 'RED';
    if (blocks.length >= 1 || alerts.length >= 3) return 'ORANGE';
    if (alerts.length >= 2) return 'YELLOW';
    return 'GREEN';
  }, [incidents]);

  const alertLevel = calculateAlertLevel();
  const blockedToday = incidents.filter(i => i.action === 'block').length;
  const activeAlerts = incidents.filter(i => i.action === 'alert').length;
  const uniqueVictims = new Set(incidents.filter(i => i.action !== 'allow').map(i => i.receiver));

  useEffect(() => {
    const unsub = subscribeToIncidents((incident) => {
      setIncidents(prev => [incident, ...prev]);
      setUnreadCount(prev => prev + 1);

      if (incident.action === 'block') {
        setBlockToast(incident);
        // Play chime
        try {
          const audio = new Audio(CHIME_AUDIO);
          audio.volume = 0.3;
          audio.play().catch(() => {});
        } catch (e) { /* ignore */ }
        
        // Auto-dismiss toast after 5s
        setTimeout(() => setBlockToast(null), 5000);
      }
    });
    return unsub;
  }, []);

  const renderContent = () => {
    switch (activeSection) {
      case 'feed':
        return (
          <div className="admin-content-split">
            <IncidentFeed
              incidents={incidents}
              onSelectIncident={setSelectedIncident}
              selectedId={selectedIncident?.id || null}
            />
            {selectedIncident && (
              <CaseDetail incident={selectedIncident} />
            )}
          </div>
        );
      case 'predators':
      case 'victims':
        return <UserProfiles incidents={incidents} section={activeSection} />;
      case 'health':
        return <SystemHealth />;
      case 'evidence':
        return (
          <div className="admin-section-placeholder">
            <h2>Evidence Files</h2>
            <p className="text-muted">Select an incident from the Live Feed to export evidence.</p>
          </div>
        );
      case 'analytics':
        return (
          <div className="admin-section-placeholder">
            <h2>Analytics</h2>
            <div className="analytics-grid">
              <div className="analytics-card">
                <span className="analytics-value font-mono">{incidents.length}</span>
                <span className="analytics-label">Total Incidents</span>
              </div>
              <div className="analytics-card">
                <span className="analytics-value font-mono text-red">{blockedToday}</span>
                <span className="analytics-label">Messages Blocked</span>
              </div>
              <div className="analytics-card">
                <span className="analytics-value font-mono text-amber">{activeAlerts}</span>
                <span className="analytics-label">Alerts Issued</span>
              </div>
              <div className="analytics-card">
                <span className="analytics-value font-mono text-cyan">{uniqueVictims.size}</span>
                <span className="analytics-label">Protected Users</span>
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="admin-page theme-admin">
      <AdminHeader
        alertLevel={alertLevel}
        activeAlerts={activeAlerts}
        blockedToday={blockedToday}
        protectedUsers={uniqueVictims.size}
      />
      <div className="admin-body">
        <AdminSidebar
          activeSection={activeSection}
          onSectionChange={(s) => {
            setActiveSection(s);
            setUnreadCount(0);
          }}
          unreadCount={unreadCount}
        />
        <main className="admin-main">
          {renderContent()}
        </main>
      </div>
      {blockToast && (
        <BlockToast
          incident={blockToast}
          onDismiss={() => setBlockToast(null)}
          onViewCase={() => {
            setSelectedIncident(blockToast);
            setActiveSection('feed');
            setBlockToast(null);
          }}
        />
      )}
    </div>
  );
}
