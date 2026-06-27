/**
 * App.jsx
 */

import { useChat } from './hooks/useChat'
import UploadPanel from './components/UploadPanel'
import ChatWindow from './components/ChatWindow'
import MessageInput from './components/MessageInput'
import styles from './App.module.css'

export default function App() {
  const {
    messages,
    isLoading,
    isPdfLoaded,
    pdfName,
    uploadStatus,
    handleUpload,
    handleSend,
  } = useChat()

  return (
    <div className={styles.app}>
      {/* ── Sidebar ────────────────────────────────────────────────────── */}
      <aside className={styles.sidebar}>
        {/* Branding */}
        <div className={styles.brand}>
          <div className={styles.brandIcon} aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <path d="M9 15h6M9 11h6" />
            </svg>
          </div>
          <div>
            <h1 className={styles.brandName}>PDF Chat</h1>
            <p className={styles.brandTagline}>RAG-powered Q&amp;A</p>
          </div>
        </div>

        <div className={styles.sidebarDivider} />

        {/* Upload section */}
        <div className={styles.sidebarSection}>
          <p className={styles.sectionLabel}>Document</p>
          <UploadPanel
            uploadStatus={uploadStatus}
            onUpload={handleUpload}
            isPdfLoaded={isPdfLoaded}
          />
        </div>

        {/* Active document info */}
        {isPdfLoaded && pdfName && (
          <div className={styles.docInfo}>
            <div className={styles.docIcon} aria-hidden="true">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className={styles.docName}>{pdfName}</span>
          </div>
        )}

        <div className={styles.sidebarSpacer} />

        {/* Footer */}
        <div className={styles.sidebarFooter}>
          <p className={styles.footerText}>
            Built with FastAPI · LangChain · Pinecone · Groq
          </p>
        </div>
      </aside>

      {/* ── Chat panel ─────────────────────────────────────────────────── */}
      <main className={styles.chatPanel}>
        <ChatWindow messages={messages} isLoading={isLoading} />
        <MessageInput
          onSend={handleSend}
          isLoading={isLoading}
          isPdfLoaded={isPdfLoaded}
        />
      </main>
    </div>
  )
}
