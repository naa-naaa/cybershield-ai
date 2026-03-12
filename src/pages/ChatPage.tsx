import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCurrentUser, getOtherUsers, getMessages, sendMessage, subscribeToMessages, createIncident, generateNotification } from '../lib/store';
import { analyzeMessage } from '../lib/analyze';
import { Message, AnalyzeResponse, Notification } from '../lib/types';
import ChatSidebar from '../components/chat/ChatSidebar';
import ChatThread from '../components/chat/ChatThread';
import ChatInput from '../components/chat/ChatInput';
import ChatHeader from '../components/chat/ChatHeader';
import AlertBanner from '../components/chat/AlertBanner';
import BlockOverlay from '../components/chat/BlockOverlay';
import NotificationDropdown from '../components/chat/NotificationDropdown';
import './ChatPage.css';

export default function ChatPage() {
  const navigate = useNavigate();
  const currentUser = getCurrentUser();
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [shieldOn, setShieldOn] = useState(false);
  const [shieldOffline, setShieldOffline] = useState(false);
  const [strikeCount, setStrikeCount] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [showNotifications, setShowNotifications] = useState(false);
  
  // Alert banner state
  const [alertData, setAlertData] = useState<{
    response: AnalyzeResponse;
    text: string;
  } | null>(null);
  
  // Block overlay state
  const [blockData, setBlockData] = useState<{
    response: AnalyzeResponse;
    strikeNum: number;
  } | null>(null);

  // Pending message for alert flow
  const pendingMessageRef = useRef<{ text: string; receiver: string } | null>(null);

  useEffect(() => {
    if (!currentUser) {
      navigate('/login');
      return;
    }
    const others = getOtherUsers(currentUser);
    if (others.length > 0 && !activeChat) {
      setActiveChat(others[0]);
    }
  }, [currentUser, navigate, activeChat]);

  // Load messages when active chat changes
  useEffect(() => {
    if (!currentUser || !activeChat) return;
    
    const loadMessages = async () => {
      const msgs = await getMessages(currentUser, activeChat);
      setMessages(msgs);
    };
    loadMessages();
  }, [currentUser, activeChat]);

  // Subscribe to real-time messages
  useEffect(() => {
    if (!currentUser || !activeChat) return;
    
    const unsub = subscribeToMessages(async () => {
      const msgs = await getMessages(currentUser, activeChat);
      setMessages(msgs);
    });
    return unsub;
  }, [currentUser, activeChat]);

  // Cooldown timer
  useEffect(() => {
    if (cooldown <= 0) return;
    const interval = setInterval(() => {
      setCooldown(prev => {
        if (prev <= 1) {
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [cooldown]);

  const handleSend = useCallback(async (text: string, fileUrl?: string, fileName?: string) => {
    if (!currentUser || !activeChat || cooldown > 0) return;
    
    const timestamp = new Date().toISOString();
    
    if (shieldOn) {
      setAnalyzing(true);
      setShieldOffline(false);
      
      const threadHistory = messages.slice(-10).map(m => m.content);
      
      const response = await analyzeMessage({
        text,
        sender: currentUser,
        receiver: activeChat,
        thread_id: `conv_${currentUser}_${activeChat}`,
        timestamp,
        thread_history: threadHistory,
      });

      setAnalyzing(false);

      if (!response) {
        // Timeout — allow silently, show offline pill
        setShieldOffline(true);
        setTimeout(() => setShieldOffline(false), 5000);
        
        await sendMessage({
          sender: currentUser,
          receiver: activeChat,
          content: text,
          status: 'sent',
          harm_score: null,
          category: null,
          timestamp,
          file_url: fileUrl,
          file_name: fileName,
        });
        return;
      }

      // Write incident regardless
      await createIncident({
        sender: currentUser,
        receiver: activeChat,
        content: text,
        action: response.action,
        harm_score: response.harm_score,
        category: response.category,
        agent_t1: response.agent_scores?.t1_toxicity ?? null,
        agent_t2: response.agent_scores?.t2_context ?? null,
        agent_t3: response.agent_scores?.t3_emotion ?? null,
        timestamp,
      });

      if (response.action === 'allow') {
        await sendMessage({
          sender: currentUser,
          receiver: activeChat,
          content: text,
          status: 'sent',
          harm_score: response.harm_score,
          category: response.category,
          timestamp,
          file_url: fileUrl,
          file_name: fileName,
        });
      } else if (response.action === 'alert') {
        pendingMessageRef.current = { text, receiver: activeChat };
        setAlertData({ response, text });
      } else if (response.action === 'block') {
        const newStrike = strikeCount + 1;
        setStrikeCount(newStrike);
        setBlockData({ response, strikeNum: newStrike });
        setCooldown(30);
        
        setNotifications(prev => [
          generateNotification('block', currentUser, activeChat, response.category, newStrike),
          ...prev,
        ]);
      }
    } else {
      // Shield OFF — send directly
      await sendMessage({
        sender: currentUser,
        receiver: activeChat,
        content: text,
        status: 'sent',
        harm_score: null,
        category: null,
        timestamp,
        file_url: fileUrl,
        file_name: fileName,
      });
    }
  }, [currentUser, activeChat, shieldOn, messages, strikeCount, cooldown]);

  const handleAlertSendAnyway = useCallback(async () => {
    if (!pendingMessageRef.current || !currentUser || !alertData) return;
    
    const { text, receiver } = pendingMessageRef.current;
    const newStrike = strikeCount + 1;
    setStrikeCount(newStrike);
    
    await sendMessage({
      sender: currentUser,
      receiver,
      content: text,
      status: 'flagged',
      harm_score: alertData.response.harm_score,
      category: alertData.response.category,
      timestamp: new Date().toISOString(),
    });
    
    setNotifications(prev => [
      generateNotification('alert', currentUser, receiver, alertData.response.category, newStrike),
      ...prev,
    ]);
    
    setAlertData(null);
    pendingMessageRef.current = null;
  }, [currentUser, alertData, strikeCount]);

  const handleAlertEdit = useCallback(() => {
    setAlertData(null);
    // The text will be restored via the return value
    return pendingMessageRef.current?.text || '';
  }, []);

  const handleAlertCancel = useCallback(() => {
    setAlertData(null);
    pendingMessageRef.current = null;
  }, []);

  const handleBlockDismiss = useCallback(() => {
    setBlockData(null);
  }, []);

  if (!currentUser) return null;

  const otherUsers = getOtherUsers(currentUser);
  const unreadNotifications = notifications.filter(n => !n.read).length;

  return (
    <div className="chat-page theme-chat">
      <ChatHeader
        shieldOn={shieldOn}
        shieldOffline={shieldOffline}
        onToggleShield={() => setShieldOn(!shieldOn)}
        strikeCount={strikeCount}
        notificationCount={unreadNotifications}
        onNotificationClick={() => setShowNotifications(!showNotifications)}
        currentUser={currentUser}
      />
      
      {showNotifications && (
        <NotificationDropdown
          notifications={notifications}
          onClose={() => setShowNotifications(false)}
          onMarkRead={(id) => setNotifications(prev => 
            prev.map(n => n.id === id ? { ...n, read: true } : n)
          )}
        />
      )}

      <div className="chat-body">
        <ChatSidebar
          currentUser={currentUser}
          users={otherUsers}
          activeChat={activeChat}
          onSelectChat={setActiveChat}
          messages={messages}
        />
        
        <div className="chat-main">
          {alertData && (
            <AlertBanner
              category={alertData.response.category}
              harmScore={alertData.response.harm_score}
              onSendAnyway={handleAlertSendAnyway}
              onEdit={handleAlertEdit}
              onCancel={handleAlertCancel}
            />
          )}
          
          {activeChat ? (
            <>
              <ChatThread
                messages={messages}
                currentUser={currentUser}
              />
              <ChatInput
                onSend={handleSend}
                disabled={cooldown > 0}
                cooldown={cooldown}
                analyzing={analyzing}
                editText={alertData === null ? undefined : undefined}
              />
            </>
          ) : (
            <div className="chat-empty">
              <p>Select a conversation to start chatting</p>
            </div>
          )}
        </div>
      </div>

      {blockData && (
        <BlockOverlay
          strikeNum={blockData.strikeNum}
          category={blockData.response.category}
          cooldown={cooldown}
          onDismiss={handleBlockDismiss}
        />
      )}
    </div>
  );
}
