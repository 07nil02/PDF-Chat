/**
 * MessageInput.jsx
 * ----------------
 * The chat input bar at the bottom of the interface.
 *
 * Behaviours:
 *   - Textarea auto-grows up to 5 lines, then scrolls
 *   - Enter sends the message; Shift+Enter inserts a newline
 *   - Disabled and shows a hint when no PDF is loaded yet
 *   - Disabled during API loading to prevent duplicate submissions
 *   - Send button activates only when there is non-whitespace content
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import styles from './MessageInput.module.css'

export default function MessageInput({ onSend, isLoading, isPdfLoaded }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea height based on content
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }, [value])

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim()
    if (!trimmed || isLoading || !isPdfLoaded) return
    onSend(trimmed)
    setValue('')
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }, [value, isLoading, isPdfLoaded, onSend])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  const isDisabled = isLoading || !isPdfLoaded
  const canSend = value.trim().length > 0 && !isDisabled

  const placeholder = !isPdfLoaded
    ? 'Upload a PDF to start asking questions…'
    : isLoading
    ? 'Waiting for answer…'
    : 'Ask anything about the document…'

  return (
    <div className={styles.inputBar}>
      <div className={[styles.inputWrapper, isDisabled && styles.disabled].filter(Boolean).join(' ')}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isDisabled}
          rows={1}
          aria-label="Chat message input"
          aria-disabled={isDisabled}
        />
        <button
          className={[styles.sendButton, canSend && styles.active].filter(Boolean).join(' ')}
          onClick={handleSubmit}
          disabled={!canSend}
          aria-label="Send message"
        >
          {isLoading ? <LoadingIcon /> : <SendIcon />}
        </button>
      </div>
      <p className={styles.hint}>
        {isPdfLoaded
          ? 'Enter to send · Shift+Enter for new line'
          : 'Upload a PDF above to enable chat'}
      </p>
    </div>
  )
}

function SendIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  )
}

function LoadingIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"
        style={{ animation: 'spin 1s linear infinite', transformOrigin: 'center' }}
      />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </svg>
  )
}
