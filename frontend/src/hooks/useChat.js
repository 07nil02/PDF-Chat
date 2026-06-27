/**
 * hooks/useChat.js
 * ----------------
 * Custom hook that owns all application state and API interactions.
 * Components stay purely presentational — they render state and call handlers.
 *
 * State managed here:
 *   messages     - array of { id, role, content, sources?, isError? }
 *   isLoading    - true while waiting for an API response
 *   isPdfLoaded  - true after a PDF has been successfully ingested
 *   pdfName      - filename of the currently loaded PDF
 *   uploadStatus - { type: 'idle'|'loading'|'success'|'error', message: string }
 */

import { useState, useCallback, useRef } from 'react'
import { uploadPDF, askQuestion } from '../api/client'

// Safe helper to generate unique IDs
function generateUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }

  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
}

const SESSION_ID = generateUUID()

const WELCOME_MESSAGE = {
  id: 'welcome',
  role: 'assistant',
  content: 'Upload a PDF document and I\'ll answer questions about it. I only use information from the document — no guessing.',
}

export function useChat() {
  const [messages, setMessages]       = useState([WELCOME_MESSAGE])
  const [isLoading, setIsLoading]     = useState(false)
  const [isPdfLoaded, setIsPdfLoaded] = useState(false)
  const [pdfName, setPdfName]         = useState(null)
  const [uploadStatus, setUploadStatus] = useState({ type: 'idle', message: '' })


  const historyRef = useRef([])

  /**
   * Append a message to the thread.
   * @param {{ role: string, content: string, sources?: array, isError?: bool }} msg
   */
  const addMessage = useCallback((msg) => {
    setMessages(prev => [...prev, { id: generateUUID(), ...msg }])
  }, [])

  /**
   * Handle PDF file selection (from drag-drop or file input click).
   * Sends the file to POST /upload and tracks upload state.
   * @param {File} file
   */
  const handleUpload = useCallback(async (file) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus({ type: 'error', message: 'Only PDF files are supported.' })
      return
    }

    setUploadStatus({ type: 'loading', message: `Processing ${file.name}…` })
    setIsPdfLoaded(false)
    setMessages([WELCOME_MESSAGE])
    historyRef.current = []   // clear history on new PDF

    try {
      // Pass session_id as query param so backend clears memory for this session
      const result = await uploadPDF(file, SESSION_ID)
      setPdfName(file.name)
      setIsPdfLoaded(true)
      setUploadStatus({
        type: 'success',
        message: `${file.name} — ${result.chunks} chunks indexed`,
      })
      addMessage({
        role: 'assistant',
        content: `**${file.name}** is ready (${result.chunks} chunks). Ask me anything about it.`,
      })
    } catch (err) {
      setUploadStatus({ type: 'error', message: err.message })
      setIsPdfLoaded(false)
    }
  }, [addMessage])

  /**
   * Send a user question to POST /chat and add the response to the thread.
   * @param {string} question
   */
  const handleSend = useCallback(async (question) => {
    if (!question.trim() || isLoading) return

    addMessage({ role: 'user', content: question })
    setIsLoading(true)

    try {
      const result = await askQuestion(question, SESSION_ID, historyRef.current)
      addMessage({
        role: 'assistant',
        content: result.answer,
        sources: result.sources || [],
      })
      // Update local history ref
      historyRef.current = [
        ...historyRef.current,
        { role: 'user',      content: question       },
        { role: 'assistant', content: result.answer  },
      ]
    } catch (err) {
      addMessage({ role: 'assistant', content: `Error: ${err.message}`, isError: true })
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, addMessage])

  return {
    messages,
    isLoading,
    isPdfLoaded,
    pdfName,
    uploadStatus,
    handleUpload,
    handleSend,
  }
}
