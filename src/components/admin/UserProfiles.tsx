import { ShieldAlert, ShieldCheck, UserCheck, Ban, Eye, FileDown } from 'lucide-react';
import { Incident } from '../../lib/types';
import './UserProfiles.css';

interface UserProfilesProps {
  incidents: Incident[];
  section: 'predators' | 'victims';
}

interface UserStats {
  username: string;
  sentFlagged: number;
  receivedFlagged: number;
  targets: Set<string>;
  attackers: Set<string>;
  strikeCount: number;
  lastIncidentTime: string;
}

export default function UserProfiles({ incidents, section }: UserProfilesProps) {
  // Aggregate user stats from incidents
  const userMap = new Map<string, UserStats>();
  const allUsers = new Set<string>();

  incidents.forEach(i => {
    allUsers.add(i.sender);
    allUsers.add(i.receiver);
  });

  allUsers.forEach(username => {
    const stats: UserStats = {
      username,
      sentFlagged: 0,
      receivedFlagged: 0,
      targets: new Set(),
      attackers: new Set(),
      strikeCount: 0,
      lastIncidentTime: '',
    };

    incidents.forEach(i => {
      if (i.sender === username && i.action !== 'allow') {
        stats.sentFlagged++;
        stats.targets.add(i.receiver);
        if (i.action === 'block') stats.strikeCount++;
        if (!stats.lastIncidentTime || i.timestamp > stats.lastIncidentTime) {
          stats.lastIncidentTime = i.timestamp;
        }
      }
      if (i.receiver === username && i.action !== 'allow') {
        stats.receivedFlagged++;
        stats.attackers.add(i.sender);
      }
    });

    userMap.set(username, stats);
  });

  const users = Array.from(userMap.values());
  
  const predators = users.filter(u => u.sentFlagged > 0).sort((a, b) => b.sentFlagged - a.sentFlagged);
  const victims = users.filter(u => u.receivedFlagged > 0).sort((a, b) => b.receivedFlagged - a.receivedFlagged);
  const safeUsers = users.filter(u => u.sentFlagged === 0 && u.receivedFlagged === 0);

  if (section === 'predators') {
    return (
      <div className="user-profiles">
        <h2>Predator Profiles</h2>
        {predators.length === 0 ? (
          <p className="text-muted up-empty">No predator profiles detected yet.</p>
        ) : (
          <div className="profile-grid">
            {predators.map(user => (
              <div key={user.username} className="profile-card predator">
                <div className="profile-card-header">
                  <div className="profile-avatar predator">
                    <ShieldAlert size={20} />
                  </div>
                  <div className="profile-info">
                    <span className="profile-name">{user.username}</span>
                    <span className="profile-badge predator">PREDATOR RISK</span>
                  </div>
                  <span className="profile-strikes font-mono">
                    Strikes: {user.strikeCount}/3
                  </span>
                </div>
                <div className="profile-stats">
                  <span>Flagged sent: <strong>{user.sentFlagged}</strong></span>
                  <span>Targets {Array.from(user.targets).join(', ')} often</span>
                </div>
                <div className="profile-actions">
                  <button className="prof-btn prof-btn-ban"><Ban size={13} /> Block Now</button>
                  <button className="prof-btn"><Eye size={13} /> View All Messages</button>
                  <button className="prof-btn"><FileDown size={13} /> Export</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="user-profiles">
      <h2>Victim Protection</h2>
      {victims.length === 0 && safeUsers.length === 0 ? (
        <p className="text-muted up-empty">No user profiles to display yet.</p>
      ) : (
        <div className="profile-grid">
          {victims.map(user => (
            <div key={user.username} className="profile-card victim">
              <div className="profile-card-header">
                <div className="profile-avatar victim">
                  <ShieldCheck size={20} />
                </div>
                <div className="profile-info">
                  <span className="profile-name">{user.username}</span>
                  <span className="profile-badge victim">VICTIM · Protected</span>
                </div>
              </div>
              <div className="profile-stats">
                <span>Harmful msgs blocked: <strong>{user.receivedFlagged}</strong></span>
                {user.lastIncidentTime && (
                  <span>Last: {getTimeAgo(user.lastIncidentTime)}</span>
                )}
              </div>
              <div className="profile-actions">
                <button className="prof-btn prof-btn-safe"><ShieldCheck size={13} /> Enable Safe Mode</button>
                <button className="prof-btn">Send Resources</button>
                <button className="prof-btn"><FileDown size={13} /> Evidence</button>
              </div>
            </div>
          ))}
          {safeUsers.map(user => (
            <div key={user.username} className="profile-card safe">
              <div className="profile-card-header">
                <div className="profile-avatar safe">
                  <UserCheck size={20} />
                </div>
                <div className="profile-info">
                  <span className="profile-name">{user.username}</span>
                  <span className="profile-badge safe">SAFE ✓</span>
                </div>
              </div>
              <div className="profile-stats">
                <span>No flags. 0 incidents.</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function getTimeAgo(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  return `${Math.floor(minutes / 60)}h ago`;
}
