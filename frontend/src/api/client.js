/**
 * Retrieva AI - HTTP API Client
 *
 * Provides typed, error-handled HTTP abstractions for interacting with the FastAPI backend.
 */

const API_BASE_URL = 
  (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  (typeof process !== 'undefined' && process.env && process.env.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8000/api/v1';

/**
 * Custom API Error carrying HTTP status code and user-facing detail.
 */
export class ApiError extends Error {
  constructor(message, status = 500, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Handle HTTP response parsing and throw structured ApiError on failure.
 */
async function handleResponse(response) {
  if (response.ok) {
    return await response.json();
  }

  let errorDetail = '';
  try {
    const errorJson = await response.json();
    errorDetail = errorJson.detail || errorJson.message || JSON.stringify(errorJson);
  } catch {
    errorDetail = await response.text() || response.statusText;
  }

  let userFriendlyMessage = `Server error (${response.status}): ${errorDetail}`;
  if (response.status === 404) {
    userFriendlyMessage = `Resource not found: ${errorDetail}`;
  } else if (response.status === 400) {
    userFriendlyMessage = `Invalid request: ${errorDetail}`;
  } else if (response.status === 413) {
    userFriendlyMessage = `File too large: ${errorDetail}`;
  } else if (response.status === 429) {
    userFriendlyMessage = `Rate limit exceeded. Please wait a moment before trying again.`;
  } else if (response.status === 502 || response.status === 503 || response.status === 504) {
    userFriendlyMessage = `Backend service unavailable: ${errorDetail}`;
  }

  throw new ApiError(userFriendlyMessage, response.status, errorDetail);
}

/**
 * Fetch health status of the FastAPI backend.
 */
export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Cannot connect to Retrieva AI backend server. Ensure uvicorn is running.', 0);
  }
}

/**
 * Fetch list of all ingested documents in the system.
 */
export async function fetchDocuments() {
  try {
    const res = await fetch(`${API_BASE_URL}/documents`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Failed to fetch document catalog from backend.', 0);
  }
}

/**
 * Upload a PDF or TXT document file for ingestion, chunking, and embedding.
 *
 * @param {File} file - The file to upload
 * @returns {Promise<Object>} DocumentUploadResponse
 */
export async function uploadDocument(file) {
  if (!file) {
    throw new ApiError('No file provided for upload.', 400);
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Network error while uploading '${file.name}': ${err.message}`, 0);
  }
}

/**
 * Delete an ingested document by its document_id.
 *
 * @param {string} documentId - The unique ID of the document to delete
 * @returns {Promise<Object>} DocumentDeleteResponse
 */
export async function deleteDocument(documentId) {
  if (!documentId) {
    throw new ApiError('Document ID is required for deletion.', 400);
  }

  try {
    const res = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}`, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Failed to delete document '${documentId}': ${err.message}`, 0);
  }
}

/**
 * Ask a question to the RAG pipeline (non-streaming).
 *
 * @param {Object} params
 * @param {string} params.query - The natural language question
 * @param {string} [params.retrieval_mode="hybrid"] - "dense", "bm25", or "hybrid"
 * @param {number} [params.top_k=5] - Number of candidate chunks
 * @param {string[]} [params.document_ids=null] - Optional document filter
 * @returns {Promise<Object>} AnswerResponse
 */
export async function askQuestion({
  query,
  retrieval_mode = 'hybrid',
  top_k = 5,
  document_ids = null,
  conversation_id = null,
}) {
  const cleanQuery = (query || '').trim();
  if (!cleanQuery) {
    throw new ApiError('Query string cannot be empty.', 400);
  }

  const payload = {
    query: cleanQuery,
    document_ids: Array.isArray(document_ids) && document_ids.length > 0 ? document_ids : null,
    conversation_id: conversation_id || null,
    retrieval_config: {
      mode: retrieval_mode.toLowerCase(),
      top_k: Number(top_k) || 5,
      candidate_k: 20,
      enable_reranking: true,
      similarity_threshold: 0.0,
      rrf_k: 60,
    },
  };

  try {
    const res = await fetch(`${API_BASE_URL}/answer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error communicating with RAG answer generation service: ${err.message}`, 0);
  }
}

/**
 * Execute standalone retrieval search against the document knowledge base without LLM generation.
 *
 * @param {Object} params
 * @param {string} params.query - The search query
 * @param {string} [params.mode="hybrid"] - "dense", "bm25", or "hybrid"
 * @param {number} [params.top_k=5] - Number of candidate chunks to return
 * @param {number} [params.candidate_k=20] - Number of candidate chunks to collect before fusion/reranking
 * @param {boolean} [params.enable_reranking=false] - Whether to apply Cross-Encoder reranking
 * @param {number|null} [params.similarity_threshold=null] - Optional minimum similarity threshold for dense mode
 * @param {string|null} [params.document_id_filter=null] - Optional document filter
 * @returns {Promise<Object>} RetrievalResult
 */
export async function searchRetrieval({
  query,
  mode = 'hybrid',
  top_k = 5,
  candidate_k = 20,
  enable_reranking = false,
  similarity_threshold = null,
  document_id_filter = null,
} = {}) {
  const cleanQuery = (query || '').trim();
  if (!cleanQuery) {
    throw new ApiError('Query string cannot be empty.', 400);
  }

  const payload = {
    query: cleanQuery,
    config: {
      mode: (mode || 'hybrid').toLowerCase(),
      top_k: Number(top_k) || 5,
      candidate_k: Math.max(Number(candidate_k) || 20, Number(top_k) || 5),
      enable_reranking: Boolean(enable_reranking),
      similarity_threshold: similarity_threshold !== null ? Number(similarity_threshold) : null,
      document_id_filter: document_id_filter || null,
    },
  };

  try {
    const res = await fetch(`${API_BASE_URL}/retrieval/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error communicating with retrieval service: ${err.message}`, 0);
  }
}

/**
 * Stream an answer incrementally from the RAG pipeline via Server-Sent Events (SSE).
 *
 * Emits progressive token updates, intermediate stage status events, and safe post-stream
 * citation validation results. If citation verification fails, an 'onRejection' callback
 * is fired so the UI can discard provisional tokens and replace them with a safe fallback.
 *
 * @param {Object} params
 * @param {string} params.query - The natural language question
 * @param {string} [params.retrieval_mode="hybrid"] - "dense", "bm25", or "hybrid"
 * @param {number} [params.top_k=5] - Number of candidate chunks
 * @param {string[]} [params.document_ids=null] - Optional document filter
 * @param {Function} [params.onStatus] - Callback receiving status payloads ({stage, message})
 * @param {Function} [params.onToken] - Callback receiving token text delta (delta: string)
 * @param {Function} [params.onCitations] - Callback receiving citations & metadata ({status, citations, insufficient_context, metadata, ttft_ms})
 * @param {Function} [params.onRejection] - Callback receiving citation rejection ({status, reason, invalid_citations, safe_fallback})
 * @param {Function} [params.onError] - Callback receiving error payload ({code, message})
 * @param {Function} [params.onDone] - Callback receiving stream completion notification ({finished: true})
 * @param {AbortSignal} [params.signal] - AbortSignal for user cancellation
 * @returns {Promise<void>}
 */
export async function askQuestionStream({
  query,
  retrieval_mode = 'hybrid',
  top_k = 5,
  document_ids = null,
  conversation_id = null,
  onStatus = () => {},
  onToken = () => {},
  onCitations = () => {},
  onRejection = () => {},
  onError = () => {},
  onDone = () => {},
  signal = null,
} = {}) {
  const cleanQuery = (query || '').trim();
  if (!cleanQuery) {
    const err = new ApiError('Query string cannot be empty.', 400);
    onError({ code: 'invalid_request', message: err.message });
    throw err;
  }

  const payload = {
    query: cleanQuery,
    document_ids: Array.isArray(document_ids) && document_ids.length > 0 ? document_ids : null,
    conversation_id: conversation_id || null,
    retrieval_config: {
      mode: (retrieval_mode || 'hybrid').toLowerCase(),
      top_k: Number(top_k) || 5,
      candidate_k: 20,
      enable_reranking: true,
      similarity_threshold: 0.0,
      rrf_k: 60,
    },
  };

  let res;
  try {
    res = await fetch(`${API_BASE_URL}/answer/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (fetchErr) {
    if (signal?.aborted || fetchErr.name === 'AbortError') {
      return;
    }
    const err = new ApiError(`Network error while initiating stream: ${fetchErr.message}`, 0);
    onError({ code: 'network_error', message: err.message });
    throw err;
  }

  if (!res.ok) {
    let errorDetail = '';
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || JSON.stringify(errJson);
    } catch {
      errorDetail = (await res.text()) || res.statusText;
    }
    const apiErr = new ApiError(`Streaming request failed (${res.status}): ${errorDetail}`, res.status, errorDetail);
    onError({ code: 'http_error', message: apiErr.message });
    throw apiErr;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    const err = new ApiError('Response body is not readable for SSE streaming.', 500);
    onError({ code: 'stream_unavailable', message: err.message });
    throw err;
  }

  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  const processFrame = (rawFrame) => {
    if (!rawFrame.trim()) return;
    let eventType = 'message';
    const dataLines = [];

    for (const line of rawFrame.split('\n')) {
      const trimmed = line.trimEnd();
      if (trimmed.startsWith('event:')) {
        eventType = trimmed.slice(6).trim();
      } else if (trimmed.startsWith('data:')) {
        dataLines.push(trimmed.slice(5).trim());
      }
    }

    const rawData = dataLines.join('\n');
    let parsedData = rawData;
    try {
      parsedData = JSON.parse(rawData);
    } catch {
      // Keep string if not valid JSON
    }

    switch (eventType) {
      case 'status':
        onStatus(parsedData);
        break;
      case 'token':
        onToken(parsedData.delta !== undefined ? parsedData.delta : parsedData);
        break;
      case 'citations':
        onCitations(parsedData);
        break;
      case 'rejection':
        onRejection(parsedData);
        break;
      case 'error':
        onError(parsedData);
        break;
      case 'done':
        onDone(parsedData);
        break;
      default:
        break;
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';

      for (const frame of frames) {
        processFrame(frame);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      processFrame(buffer);
    }
  } catch (readErr) {
    if (signal?.aborted || readErr.name === 'AbortError') {
      return;
    }
    const err = new ApiError(`Error while reading event stream: ${readErr.message}`, 0);
    onError({ code: 'stream_read_error', message: err.message });
    throw err;
  } finally {
    reader.releaseLock?.();
  }
}

/**
 * Create a new persistent conversation.
 *
 * @param {string|Object|null} [titleOrOptions=null] - Optional initial conversation title or { title }
 * @returns {Promise<Object>} ConversationRecord
 */
export async function createConversation(titleOrOptions = null) {
  let title = null;
  if (typeof titleOrOptions === 'string') {
    title = titleOrOptions;
  } else if (titleOrOptions && typeof titleOrOptions === 'object' && titleOrOptions.title) {
    title = titleOrOptions.title;
  }

  try {
    const res = await fetch(`${API_BASE_URL}/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(title ? { title } : {}),
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error creating conversation: ${err.message}`, 0);
  }
}

/**
 * Fetch all persistent conversations ordered by updated_at descending.
 *
 * @param {number} [limit=50] - Maximum conversations to return
 * @param {number} [offset=0] - Pagination offset
 * @returns {Promise<Array<Object>>} List of ConversationRecords
 */
export async function fetchConversations(limit = 50, offset = 0) {
  try {
    const res = await fetch(`${API_BASE_URL}/conversations?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error fetching conversations: ${err.message}`, 0);
  }
}

/**
 * Retrieve a single conversation and its full chronological message history.
 *
 * @param {string} conversationId
 * @returns {Promise<Object>} ConversationDetail
 */
export async function fetchConversation(conversationId) {
  const cleanId = (conversationId || '').trim();
  if (!cleanId) {
    throw new ApiError('Conversation ID cannot be empty.', 400);
  }
  try {
    const res = await fetch(`${API_BASE_URL}/conversations/${cleanId}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error fetching conversation '${conversationId}': ${err.message}`, 0);
  }
}

/**
 * Delete a conversation and cascade deletion to associated messages.
 *
 * @param {string} conversationId
 * @returns {Promise<Object>}
 */
export async function deleteConversation(conversationId) {
  const cleanId = (conversationId || '').trim();
  if (!cleanId) {
    throw new ApiError('Conversation ID cannot be empty.', 400);
  }
  try {
    const res = await fetch(`${API_BASE_URL}/conversations/${cleanId}`, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error deleting conversation '${conversationId}': ${err.message}`, 0);
  }
}
