import { MessageSquare } from 'lucide-react';
import { getLastMessage } from '../../lib/store';
import { Message } from '../../lib/types';
import './ChatSidebar.css';

interface ChatSidebarProps {
  currentUser: string;
  users: string[];
  activeChat: string | null;
  onSelectChat: (user: string) => void;
  messages: Message[];
}

export default function ChatSidebar({
  currentUser,
  users,
  activeChat,
  onSelectChat,
}: ChatSidebarProps) {
  return (
    <aside className="chat-sidebar">
      <div className="sidebar-header">
        <h3>Conversations</h3>
      </div>
      <div className="sidebar-list">
        {users.map(user => {
          const lastMsg = getLastMessage(currentUser, user);
          const isActive = user === activeChat;
          
          return (
            <button
              key={user}
              className={`sidebar-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelectChat(user)}
            >
              <div className="sidebar-avatar">
                {user.charAt(0).toUpperCase()}
              </div>
              <div className="sidebar-info">
                <div className="sidebar-name">{user}</div>
                <div className="sidebar-preview">
                  {lastMsg ? (
                    <>
                      <span className="preview-text">
                        {lastMsg.sender === currentUser ? 'You: ' : ''}
                        {lastMsg.content.length > 30
                          ? lastMsg.content.slice(0, 30) + '…'
                          : lastMsg.content}
                      </span>
                    </>
                  ) : (
                    <span className="preview-text text-muted">No messages yet</span>
                  )}
                </div>
              </div>
              {lastMsg && (
                <span className="sidebar-time">
                  {new Date(lastMsg.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
