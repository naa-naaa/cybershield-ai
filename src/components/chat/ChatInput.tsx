import React, { useState, useRef } from 'react';
import { Send, Paperclip, Loader2, X, Image as ImageIcon, FileText } from 'lucide-react';
import './ChatInput.css';

interface ChatInputProps {
  onSend: (text: string, fileUrl?: string, fileName?: string) => void;
  disabled: boolean;
  cooldown: number;
  analyzing: boolean;
  editText?: string;
}

export default function ChatInput({
  onSend,
  disabled,
  cooldown,
  analyzing,
  editText,
}: ChatInputProps) {
  const [text, setText] = useState(editText || '');
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileDataUrl, setFileDataUrl] = useState<string | null>(null);
  const [fileType, setFileType] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if ((!trimmed && !fileDataUrl) || disabled || analyzing) return;
    onSend(trimmed || (fileName ?? 'Shared a file'), fileDataUrl ?? undefined, fileName ?? undefined);
    setText('');
    clearFile();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const clearFile = () => {
    setFileName(null);
    setFileDataUrl(null);
    setFileType(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Limit to 5MB for BroadcastChannel
    if (file.size > 5 * 1024 * 1024) {
      alert('File size must be under 5 MB');
      return;
    }

    setFileName(file.name);
    setFileType(file.type);

    const reader = new FileReader();
    reader.onload = () => {
      setFileDataUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const isImage = fileType?.startsWith('image/');

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      {disabled && cooldown > 0 && (
        <div className="cooldown-bar">
          <span className="cooldown-text font-mono">
            Input disabled · Cooldown: {String(Math.floor(cooldown / 60)).padStart(2, '0')}:{String(cooldown % 60).padStart(2, '0')}
          </span>
        </div>
      )}

      {fileDataUrl && (
        <div className="file-preview">
          <button type="button" className="file-preview-close" onClick={clearFile}>
            <X size={16} />
          </button>
          {isImage ? (
            <img src={fileDataUrl} alt={fileName || 'Preview'} className="file-preview-image" />
          ) : (
            <div className="file-preview-doc">
              <FileText size={24} />
              <span className="file-preview-name">{fileName}</span>
            </div>
          )}
        </div>
      )}
      
      <div className={`input-row ${disabled ? 'input-disabled' : ''}`}>
        <button
          type="button"
          className="attach-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          aria-label="Attach file"
        >
          {fileDataUrl ? <ImageIcon size={18} /> : <Paperclip size={18} />}
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          accept="image/*,.pdf,.doc,.docx,.txt,.mp4,.webm"
        />
        
        <input
          type="text"
          className="message-input"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? 'Input disabled during cooldown...' : fileDataUrl ? 'Add a caption...' : 'Type a message...'}
          disabled={disabled}
          autoFocus
        />
        
        <button
          type="submit"
          className="send-btn"
          disabled={disabled || (!text.trim() && !fileDataUrl) || analyzing}
          aria-label="Send message"
        >
          {analyzing ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              <span className="send-label">Analyzing...</span>
            </>
          ) : (
            <Send size={18} />
          )}
        </button>
      </div>
    </form>
  );
}
