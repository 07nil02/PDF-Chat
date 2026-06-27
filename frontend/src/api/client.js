/**
 * api/client.js
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Upload a PDF file to the backend for ingestion.
 *
 * @param {File} file - The PDF File object from the file input / drag-drop.
 * @param {function} onProgress - Optional callback(percent: number) for future progress UI.
 * @returns {Promise<{ message: string, chunks: number }>}
 */
export async function uploadPDF(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

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
 * @returns {Promise<{ answer: string, sources: Array<{ text: string, score: number, page: number }> }>}
 */
export async function askQuestion(question) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
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
