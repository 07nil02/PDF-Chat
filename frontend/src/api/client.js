/**
 * api/client.js
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function uploadPDF(file, sessionId) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(
    `${BASE_URL}/upload?session_id=${sessionId}`,
    { method: 'POST', body: formData }
  )
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || `Upload failed with status ${response.status}`)
  }
  return response.json()
}

/**
 * Send a chat question to the backend and get an answer.
 *
 * @param {string} question - The user's question string.
 * @param {string} sessionId - The conversation session ID.
 * @returns {Promise<{ answer: string, sources: Array<{ text: string, score: number, page: number }> }>}
 */
export async function askQuestion(question, sessionId) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `Chat failed with status ${response.status}`)
  }
  return response.json()
}

/**
 * Ping the health endpoint — useful for checking backend connectivity.
 * @returns {Promise<{ status: string }>}
 */
export async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`)
  if (!response.ok) throw new Error('Backend unreachable')
  return response.json()
}

export async function checkEvalScores() {
  const response = await fetch(`${BASE_URL}/eval/scores`)
  if (!response.ok) throw new Error('Eval scores unavailable')
  return response.json()
}

export async function runEvaluation(question, answer, contextChunks) {
  const response = await fetch(`${BASE_URL}/eval/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      answer,
      context_chunks: contextChunks,
    }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Evaluation failed' }))
    throw new Error(error.detail || `Evaluation failed with status ${response.status}`)
  }
  return response.json()
}
