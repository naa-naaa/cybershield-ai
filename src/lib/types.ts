export interface Message {
  id: string;
  sender: string;
  receiver: string;
  content: string;
  status: 'sent' | 'flagged' | 'blocked';
  harm_score: number | null;
  category: string | null;
  timestamp: string;
  file_url?: string;
  file_name?: string;
}

export interface User {
  username: string;
  strike_count: number;
  status: 'active' | 'warned' | 'restricted';
}

export interface Incident {
  id: string;
  sender: string;
  receiver: string;
  content: string;
  action: 'allow' | 'alert' | 'block';
  harm_score: number;
  category: string;
  agent_t1: number | null;
  agent_t2: number | null;
  agent_t3: number | null;
  timestamp: string;
  ip_address?: string;
}

export interface AnalyzeRequest {
  text: string;
  sender: string;
  receiver: string;
  thread_id: string;
  timestamp: string;
  thread_history: string[];
}

export interface AnalyzeResponse {
  action: 'allow' | 'alert' | 'block';
  harm_score: number;
  category: string;
  agent_scores?: {
    t1_toxicity: number;
    t2_context: number;
    t3_emotion: number;
  };
}

export interface Notification {
  id: string;
  message: string;
  type: 'flagged' | 'blocked' | 'strike';
  timestamp: string;
  read: boolean;
}

export type AlertLevel = 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED';

export type AdminSection = 'feed' | 'predators' | 'victims' | 'analytics' | 'evidence' | 'health';
