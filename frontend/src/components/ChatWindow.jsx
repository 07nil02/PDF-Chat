/**
 * ChatWindow.jsx
 * --------------
 * Renders the scrollable message thread.
 *
 * Each message is one of:
 *   - user message   : right-aligned, accent background
 *   - assistant msg  : left-aligned, surface background
 *   - error message  : left-aligned, red tint
 *
 * Assistant messages render markdown via react-markdown so the LLM can use
 * bold, lists, and code blocks in its answers.
 *
 * Source chunks are shown in a collapsed <details> below each assistant
 * message — users can expand them to see exactly which part of the document
 * the answer came from. This is the "transparency" feature of RAG UIs.
 *
 * Auto-scroll: useEffect watches messages.length and scrolls the bottom
 * sentinel div into view whenever a new message is added.
 */

import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './ChatWindow.module.css'

export default function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null)

  // Auto-scroll to the latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  return (
    <div className={styles.window} role="log" aria-label="Chat messages" aria-live="polite">
      {messages.map((msg) => (
        <Message key={msg.id} message={msg} />
      ))}

      {/* Typing indicator while waiting for a response */}
      {isLoading && <TypingIndicator />}

      {/* Invisible sentinel div for auto-scroll */}
      <div ref={bottomRef} aria-hidden="true" />
    </div>
  )
}

/* ─── Individual message ─────────────────────────────────────────────────── */

function Message({ message }) {
  const isUser = message.role === 'user'
  const hasSources = message.sources && message.sources.length > 0

  return (
    <div className={[styles.messageRow, isUser ? styles.userRow : styles.assistantRow].join(' ')}>
      {!isUser && <Avatar />}

      <div className={[
        styles.bubble,
        isUser ? styles.userBubble : styles.assistantBubble,
        message.isError && styles.errorBubble,
      ].filter(Boolean).join(' ')}>

        {/* Message content — markdown for assistant, plain text for user */}
        <div className={styles.content}>
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Collapsible source chunks — assistant messages only */}
        {hasSources && (
          <details className={styles.sources}>
            <summary className={styles.sourcesSummary}>
              {message.sources.length} source{message.sources.length > 1 ? 's' : ''} used
            </summary>
            <div className={styles.sourcesList}>
              {message.sources.map((src, i) => (
                <div key={i} className={styles.sourceChunk}>
                  <div className={styles.sourceMeta}>
                    <span className={styles.sourceLabel}>Chunk {i + 1}</span>
                    <span className={styles.sourcePage}>Page {src.page + 1}</span>
                    <span className={styles.sourceScore}>
                      {(src.score * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <p className={styles.sourceText}>{src.text}</p>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

/* ─── Avatar for assistant messages ─────────────────────────────────────── */

function Avatar() {
  return (
    <div className={styles.avatar} aria-hidden="true">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/>
      </svg>
    </div>
  )
}

/* ─── Typing indicator ───────────────────────────────────────────────────── */

function TypingIndicator() {
  return (
    <div className={[styles.messageRow, styles.assistantRow].join(' ')}>
      <Avatar />
      <div className={[styles.bubble, styles.assistantBubble, styles.typingBubble].join(' ')}>
        <div className={styles.typingDots} aria-label="Assistant is typing">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

/* ─── Custom markdown renderers ──────────────────────────────────────────── */
// These map markdown AST nodes to styled HTML elements.

const markdownComponents = {
  p: ({ children }) => <p className={styles.mdP}>{children}</p>,
  strong: ({ children }) => <strong className={styles.mdStrong}>{children}</strong>,
  ul: ({ children }) => <ul className={styles.mdUl}>{children}</ul>,
  ol: ({ children }) => <ol className={styles.mdOl}>{children}</ol>,
  li: ({ children }) => <li className={styles.mdLi}>{children}</li>,
  code: ({ inline, children }) =>
    inline
      ? <code className={styles.mdInlineCode}>{children}</code>
      : <pre className={styles.mdPre}><code>{children}</code></pre>,
  blockquote: ({ children }) => <blockquote className={styles.mdBlockquote}>{children}</blockquote>,
  h1: ({ children }) => <h3 className={styles.mdH}>{children}</h3>,
  h2: ({ children }) => <h3 className={styles.mdH}>{children}</h3>,
  h3: ({ children }) => <h3 className={styles.mdH}>{children}</h3>,
}
