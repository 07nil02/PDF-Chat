import { useState, useEffect } from 'react'
import { runEvaluation } from '../api/client'
import styles from './EvalPanel.module.css'

export default function EvalPanel({ isPdfLoaded, messages }) {
  const [scores, setScores] = useState(null)
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [error, setError] = useState(null)

  // Find the last assistant message (excluding welcome) and the preceding user question
  const assistantMessages = messages.filter(m => m.role === 'assistant' && m.id !== 'welcome');
  const lastAssistant = assistantMessages[assistantMessages.length - 1];

  let lastUser = null;
  if (lastAssistant) {
    const assistantIndex = messages.findIndex(m => m.id === lastAssistant.id);
    for (let i = assistantIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUser = messages[i];
        break;
      }
    }
  }

  const hasConversation = isPdfLoaded && lastAssistant && lastUser;

  // Reset scores when a new question is asked (i.e. last user message changes)
  useEffect(() => {
    setScores(null)
    setError(null)
  }, [lastUser?.content])

  if (!hasConversation) return null;

  const handleEvaluate = async () => {
    setIsEvaluating(true)
    setError(null)
    try {
      const data = await runEvaluation(
        lastUser.content,
        lastAssistant.content,
        lastAssistant.sources || []
      )
      setScores(data)
    } catch (err) {
      setError(err.message || 'Evaluation failed')
    } finally {
      setIsEvaluating(false)
    }
  }

  const metrics = [
    { label: 'Faithfulness',      value: scores?.faithfulness,      color: getColor(scores?.faithfulness) },
    { label: 'Answer Relevancy',  value: scores?.answer_relevancy,  color: getColor(scores?.answer_relevancy) },
    { label: 'Context Precision', value: scores?.context_precision, color: getColor(scores?.context_precision) },
  ]

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.headerTitleContainer}>
          <svg className={styles.evalIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 20V10M12 20V4M6 20v-6" />
          </svg>
          <span className={styles.heading}>Quality Scores</span>
        </div>
        <span className={styles.sub}>ragas reference-free metrics</span>
      </div>

      {scores ? (
        <div className={styles.metricsContainer}>
          {metrics.map(m => {
            const val = typeof m.value === 'number' && !isNaN(m.value) ? m.value : 0;
            return (
              <div key={m.label} className={styles.metric}>
                <div className={styles.metricLabelRow}>
                  <span className={styles.label}>{m.label}</span>
                  <span className={styles.value} style={{ color: m.color }}>
                    {val.toFixed(2)}
                  </span>
                </div>
                <div className={styles.bar}>
                  <div className={styles.fill} style={{ width: `${val * 100}%`, background: m.color }} />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className={styles.placeholderContainer}>
          {error && <span className={styles.errorText}>{error}</span>}
          <p className={styles.placeholderText}>
            {isEvaluating ? 'Evaluating conversation quality...' : 'Click below to score the latest response.'}
          </p>
        </div>
      )}

      <button
        className={styles.evalButton}
        onClick={handleEvaluate}
        disabled={isEvaluating}
      >
        {isEvaluating ? (
          <>
            <svg className={styles.spinner} viewBox="0 0 24 24" fill="none">
              <circle className={styles.spinnerTrack} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className={styles.spinnerHead} d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" fill="currentColor" />
            </svg>
            Running...
          </>
        ) : (
          scores ? 'Re-Evaluate Quality' : 'Evaluate Response'
        )}
      </button>
    </div>
  )
}

function getColor(score) {
  if (score === null || score === undefined || isNaN(score)) return 'var(--text-muted)'
  if (score >= 0.8) return 'var(--success)'
  if (score >= 0.6) return '#f59e0b'
  return 'var(--error)'
}
