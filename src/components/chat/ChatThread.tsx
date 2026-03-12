import { useEffect, useRef, useState } from 'react';
import { Shield, AlertTriangle, Eye, Download, FileText } from 'lucide-react';
import { Message } from '../../lib/types';
import './ChatThread.css';

interface ChatThreadProps {
  messages: Message[];
  currentUser: string;
}

function isImageUrl(url: string): boolean {
  return url.startsWith('data:image/') || /\.(png|jpe?g|gif|webp|svg|bmp)$/i.test(url);
}

function isVideoUrl(url: string): boolean {
  return url.startsWith('data:video/') || /\.(mp4|webm|ogg)$/i.test(url);
}

export default function ChatThread({ messages, currentUser }: ChatThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [revealedMessages, setRevealedMessages] = useState<Set<string>>(new Set());

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleReveal = (id: string) => {
    setRevealedMessages(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const getConsecutiveFlaggedCount = (index: number): number => {
    if (index < 0) return 0;
    const msg = messages[index];
    if (msg.sender === currentUser) return 0;
    if (msg.status !== 'flagged') return 0;
    
    let count = 0;
    for (let i = index; i >= 0; i--) {
      if (messages[i].sender === msg.sender && messages[i].status === 'flagged') {
        count++;
      } else {
        break;
      }
    }
    return count;
  };

  const renderMedia = (msg: Message) => {
    if (!msg.file_url) return null;

    if (isImageUrl(msg.file_url)) {
      return (
        <div className="media-container">
          <img
            src={msg.file_url}
            alt={msg.file_name || 'Shared image'}
            className="media-image"
            onClick={() => window.open(msg.file_url!, '_blank')}
          />
        </div>
      );
    }

    if (isVideoUrl(msg.file_url)) {
      return (
        <div className="media-container">
          <video
            src={msg.file_url}
            controls
            className="media-video"
            preload="metadata"
          />
        </div>
      );
    }

    // Generic file
    return (
      <div className="media-file">
        <FileText size={20} />
        <div className="media-file-info">
          <span className="media-file-name">{msg.file_name || 'File'}</span>
        </div>
        <a
          href={msg.file_url}
          download={msg.file_name || 'file'}
          className="media-download-btn"
          onClick={e => e.stopPropagation()}
        >
          <Download size={16} />
        </a>
      </div>
    );
  };

  return (
    <div className="chat-thread">
      <div className="thread-messages">
        {messages.map((msg, index) => {
          const isMine = msg.sender === currentUser;
          const isFlagged = msg.status === 'flagged';
          const isRevealed = revealedMessages.has(msg.id);
          const showBlurred = !isMine && isFlagged && !isRevealed;
          const consecutiveFlagged = getConsecutiveFlaggedCount(index);
          const showSupportBanner = !isMine && isFlagged && consecutiveFlagged >= 3 && 
            (index === messages.length - 1 || messages[index + 1]?.sender !== msg.sender);

          return (
            <div key={msg.id}>
              <div
                className={`message-row ${isMine ? 'sent' : 'received'} animate-slide-up`}
              >
                <div className={`message-bubble ${isMine ? 'bubble-sent' : 'bubble-received'} ${isFlagged && isMine ? 'bubble-flagged' : ''} ${msg.file_url ? 'bubble-media' : ''}`}>
                  {showBlurred ? (
                    <div className="message-blurred">
                      <div className="blurred-content">
                        <Shield size={16} />
                        <div>
                          <p className="blurred-label">A message was hidden by CyberShield</p>
                        </div>
                        <button
                          className="reveal-btn"
                          onClick={() => toggleReveal(msg.id)}
                        >
                          <Eye size={14} />
                          View Anyway
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      {renderMedia(msg)}
                      {msg.content && (!msg.file_url || msg.content !== 'Shared a file') && (
                        <p className="message-text">{msg.content}</p>
                      )}
                      {isFlagged && isMine && (
                        <span className="flagged-icon" title="This message was flagged">
                          <AlertTriangle size={12} />
                        </span>
                      )}
                    </>
                  )}
                  <span className="message-time font-mono">
                    {new Date(msg.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span>
                </div>
              </div>

              {showSupportBanner && (
                <div className="support-banner animate-slide-up">
                  <div className="support-banner-content">
                    <p className="support-title">💙 You are not alone. Free helplines 24/7:</p>
                    <div className="support-numbers">
                      <span>iCall: <strong>9152987821</strong></span>
                      <span>Vandrevala: <strong>1860-2662-345</strong></span>
                      <span>Cybercrime Helpline: <strong>1930</strong></span>
                    </div>
                    <div className="support-actions">
                      <button className="support-btn">See Resources</button>
                      <button className="support-btn">Save Evidence →</button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
