import { Message, Incident, Notification } from './types';
import { supabase, isSupabaseConfigured } from './supabase';

// ── Local state (used when Supabase not configured) ──
let localMessages: Message[] = [];
let localIncidents: Incident[] = [];
const messageListeners: Set<() => void> = new Set();
const incidentListeners: Set<(incident: Incident) => void> = new Set();

function notifyMessageListeners() {
  messageListeners.forEach(fn => fn());
}

// ── Auth helpers ──
const VALID_USERS: Record<string, string> = {
  user1: 'user1',
  user2: 'user2',
  user3: 'user3',
};

export function login(username: string, password: string): boolean {
  if (VALID_USERS[username] && VALID_USERS[username] === password) {
    sessionStorage.setItem('cybershield_user', username);
    return true;
  }
  return false;
}

export function logout() {
  sessionStorage.removeItem('cybershield_user');
}

export function getCurrentUser(): string | null {
  return sessionStorage.getItem('cybershield_user');
}

export function getOtherUsers(currentUser: string): string[] {
  return Object.keys(VALID_USERS).filter(u => u !== currentUser);
}

// ── Message helpers ──
function getConversationKey(user1: string, user2: string): string {
  return [user1, user2].sort().join('_');
}

export async function getMessages(user1: string, user2: string): Promise<Message[]> {
  if (isSupabaseConfigured()) {
    const { data } = await supabase
      .from('messages')
      .select('*')
      .or(`and(sender.eq.${user1},receiver.eq.${user2}),and(sender.eq.${user2},receiver.eq.${user1})`)
      .order('timestamp', { ascending: true });
    return data || [];
  }
  
  return localMessages.filter(
    m => (m.sender === user1 && m.receiver === user2) ||
         (m.sender === user2 && m.receiver === user1)
  );
}

export async function sendMessage(message: Omit<Message, 'id'>): Promise<Message> {
  const newMsg: Message = {
    ...message,
    id: crypto.randomUUID(),
  };

  if (isSupabaseConfigured()) {
    const { data } = await supabase
      .from('messages')
      .insert(newMsg)
      .select()
      .single();
    return data || newMsg;
  }

  localMessages.push(newMsg);
  notifyMessageListeners();
  // Broadcast to other tabs via BroadcastChannel
  try {
    const bc = new BroadcastChannel('cybershield_messages');
    bc.postMessage({ type: 'new_message', message: newMsg });
    bc.close();
  } catch (e) { /* ignore */ }
  return newMsg;
}

export function getLastMessage(user1: string, user2: string): Message | null {
  const msgs = localMessages.filter(
    m => (m.sender === user1 && m.receiver === user2) ||
         (m.sender === user2 && m.receiver === user1)
  );
  return msgs.length > 0 ? msgs[msgs.length - 1] : null;
}

export function subscribeToMessages(callback: () => void): () => void {
  messageListeners.add(callback);
  
  // Also listen for cross-tab messages
  const bc = new BroadcastChannel('cybershield_messages');
  const handler = (event: MessageEvent) => {
    if (event.data.type === 'new_message') {
      const msg = event.data.message as Message;
      // Only add if not already present
      if (!localMessages.find(m => m.id === msg.id)) {
        localMessages.push(msg);
        callback();
      }
    }
  };
  bc.addEventListener('message', handler);
  
  return () => {
    messageListeners.delete(callback);
    bc.removeEventListener('message', handler);
    bc.close();
  };
}

// ── Incident helpers ──
export async function createIncident(incident: Omit<Incident, 'id'>): Promise<Incident> {
  const newIncident: Incident = {
    ...incident,
    id: crypto.randomUUID(),
  };

  if (isSupabaseConfigured()) {
    const { data } = await supabase
      .from('incidents')
      .insert(newIncident)
      .select()
      .single();
    return data || newIncident;
  }

  localIncidents.push(newIncident);
  incidentListeners.forEach(fn => fn(newIncident));
  // Broadcast to other tabs
  try {
    const bc = new BroadcastChannel('cybershield_incidents');
    bc.postMessage({ type: 'new_incident', incident: newIncident });
    bc.close();
  } catch (e) { /* ignore */ }
  return newIncident;
}

export function getIncidents(): Incident[] {
  return [...localIncidents].reverse();
}

export function subscribeToIncidents(callback: (incident: Incident) => void): () => void {
  incidentListeners.add(callback);
  
  const bc = new BroadcastChannel('cybershield_incidents');
  const handler = (event: MessageEvent) => {
    if (event.data.type === 'new_incident') {
      const incident = event.data.incident as Incident;
      if (!localIncidents.find(i => i.id === incident.id)) {
        localIncidents.push(incident);
        callback(incident);
      }
    }
  };
  bc.addEventListener('message', handler);
  
  return () => {
    incidentListeners.delete(callback);
    bc.removeEventListener('message', handler);
    bc.close();
  };
}

// ── Notification generation ──
export function generateNotification(
  action: 'alert' | 'block',
  sender: string,
  receiver: string,
  category: string,
  strikeCount: number
): Notification {
  let message = '';
  if (action === 'alert') {
    message = `Your message to ${receiver} was flagged for ${category}`;
  } else {
    message = `Message blocked — Strike ${strikeCount} of 3`;
  }
  return {
    id: crypto.randomUUID(),
    message,
    type: action === 'block' ? 'blocked' : 'flagged',
    timestamp: new Date().toISOString(),
    read: false,
  };
}

// ── Evidence export ──
export function generateEvidenceReport(incident: Incident): string {
  const today = new Date().toLocaleDateString('en-IN', {
    year: 'numeric', month: 'long', day: 'numeric',
  });
  return `CYBER CRIME COMPLAINT
━━━━━━━━━━━━━━━━━━━━━
Date: ${today}
Case ID: CS-${incident.id.slice(0, 8).toUpperCase()}
Platform: SafeChat Demo
Perpetrator ID: ${incident.sender}
Victim ID: ${incident.receiver}
Incident Type: ${incident.category}
Harm Score: ${incident.harm_score}
Action Taken: ${incident.action.toUpperCase()}

Message Content:
"${incident.content}"

Agent Analysis:
  T1 Toxicity:  ${incident.agent_t1?.toFixed(2) ?? 'N/A'}
  T2 Context:   ${incident.agent_t2?.toFixed(2) ?? 'N/A'}
  T3 Emotion:   ${incident.agent_t3?.toFixed(2) ?? 'N/A'}
  Final Score:  ${incident.harm_score.toFixed(2)}

Timestamp: ${new Date(incident.timestamp).toLocaleString('en-IN')}
━━━━━━━━━━━━━━━━━━━━━
Submit at: cybercrime.gov.in
Helpline: 1930
iCall: 9152987821
Vandrevala Foundation: 1860-2662-345`;
}
