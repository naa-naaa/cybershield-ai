import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../lib/store';
import { Shield, Eye, EyeOff } from 'lucide-react';
import './LoginPage.css';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    // Simulate brief auth delay
    await new Promise(r => setTimeout(r, 400));
    
    if (login(username, password)) {
      navigate('/chat');
    } else {
      setError('Invalid username or password');
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-bg-grid" />
      <div className="login-bg-glow login-glow-1" />
      <div className="login-bg-glow login-glow-2" />
      
      <div className="login-card animate-scale-in">
        <div className="login-logo">
          <div className="login-shield-icon">
            <Shield size={40} strokeWidth={2} />
          </div>
          <h1 className="login-wordmark">SafeChat</h1>
          <p className="login-subtitle">Protected by CyberShield AI</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="login-field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Enter username"
              autoComplete="username"
              autoFocus
            />
          </div>

          <div className="login-field">
            <label htmlFor="password">Password</label>
            <div className="login-password-wrap">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter password"
                autoComplete="current-password"
              />
              <button
                type="button"
                className="login-eye-btn"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {error && (
            <div className="login-error animate-fade-in">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="login-submit"
            disabled={loading || !username || !password}
          >
            {loading ? (
              <span className="login-spinner" />
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="login-footer">
          <p>Demo accounts: user1, user2, user3</p>
          <p>Password same as username</p>
        </div>
      </div>
    </div>
  );
}
