import { Radio, UserX, ShieldCheck, BarChart3, FileText, Activity } from 'lucide-react';
import { AdminSection } from '../../lib/types';
import './AdminSidebar.css';

interface AdminSidebarProps {
  activeSection: AdminSection;
  onSectionChange: (section: AdminSection) => void;
  unreadCount: number;
}

const NAV_ITEMS: { id: AdminSection; icon: React.ReactNode; label: string }[] = [
  { id: 'feed', icon: <Radio size={18} />, label: 'Live Feed' },
  { id: 'predators', icon: <UserX size={18} />, label: 'Predator Profiles' },
  { id: 'victims', icon: <ShieldCheck size={18} />, label: 'Victim Protection' },
  { id: 'analytics', icon: <BarChart3 size={18} />, label: 'Analytics' },
  { id: 'evidence', icon: <FileText size={18} />, label: 'Evidence Files' },
  { id: 'health', icon: <Activity size={18} />, label: 'System Health' },
];

export default function AdminSidebar({
  activeSection,
  onSectionChange,
  unreadCount,
}: AdminSidebarProps) {
  return (
    <aside className="admin-sidebar">
      <nav className="admin-nav">
        {NAV_ITEMS.map(item => (
          <button
            key={item.id}
            className={`admin-nav-item ${activeSection === item.id ? 'active' : ''}`}
            onClick={() => onSectionChange(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
            {item.id === 'feed' && unreadCount > 0 && (
              <span className="nav-badge">{unreadCount}</span>
            )}
          </button>
        ))}
      </nav>
    </aside>
  );
}
