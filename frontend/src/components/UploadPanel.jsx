/**
 * UploadPanel.jsx
 * ---------------
 * Drag-and-drop PDF upload zone with visual feedback for all states:
 *   idle     → dashed border, invite text
 *   dragging → accent border glow, "Drop to upload" text
 *   loading  → spinner + progress text
 *   success  → filename + chunk count
 *   error    → red border + error message
 *
 * Accepts clicks as well — clicking the zone opens the native file picker.
 */

import { useState, useRef, useCallback } from 'react'
import styles from './UploadPanel.module.css'

export default function UploadPanel({ uploadStatus, onUpload, isPdfLoaded }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDragEnter = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    // Only clear if leaving the zone entirely (not a child element)
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false)
    }
  }, [])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onUpload(file)
  }, [onUpload])

  const handleFileChange = useCallback((e) => {
    const file = e.target.files[0]
    if (file) onUpload(file)
    // Reset input so the same file can be re-uploaded if needed
    e.target.value = ''
  }, [onUpload])

  const handleClick = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      inputRef.current?.click()
    }
  }, [])

  const isLoading = uploadStatus.type === 'loading'

  return (
    <div className={styles.wrapper}>
      <div
        className={[
          styles.zone,
          isDragging && styles.dragging,
          uploadStatus.type === 'success' && styles.success,
          uploadStatus.type === 'error' && styles.error,
          isLoading && styles.loading,
        ].filter(Boolean).join(' ')}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={isLoading ? undefined : handleClick}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={isLoading ? -1 : 0}
        aria-label="Upload PDF — click or drag and drop"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className={styles.hiddenInput}
          onChange={handleFileChange}
          aria-hidden="true"
          tabIndex={-1}
        />

        {/* Idle / dragging state */}
        {uploadStatus.type === 'idle' && (
          <div className={styles.idleContent}>
            <PdfIcon className={styles.icon} />
            <span className={styles.label}>
              {isDragging ? 'Drop to upload' : 'Drop a PDF here'}
            </span>
            <span className={styles.sublabel}>or click to browse</span>
          </div>
        )}

        {/* Loading state */}
        {isLoading && (
          <div className={styles.loadingContent}>
            <Spinner />
            <span className={styles.label}>{uploadStatus.message}</span>
          </div>
        )}

        {/* Success state */}
        {uploadStatus.type === 'success' && (
          <div className={styles.successContent}>
            <CheckIcon className={styles.successIcon} />
            <span className={styles.label}>{uploadStatus.message}</span>
            <span className={styles.sublabel}>Click to swap document</span>
          </div>
        )}

        {/* Error state */}
        {uploadStatus.type === 'error' && (
          <div className={styles.errorContent}>
            <ErrorIcon className={styles.errorIcon} />
            <span className={styles.label}>{uploadStatus.message}</span>
            <span className={styles.sublabel}>Click to try again</span>
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── Inline SVG icons (no icon library needed) ─────────────────────────── */

function PdfIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10 9 9 9 8 9"/>
    </svg>
  )
}

function CheckIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  )
}

function ErrorIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  )
}

function Spinner() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"
        style={{ animation: 'spin 1s linear infinite' }}
      />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </svg>
  )
}
