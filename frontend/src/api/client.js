/**
 * Retrieva AI - Centralized HTTP API Client
 *
 * Automatically injects caller Supabase Bearer JWTs onto every protected backend
 * request, handles single-attempt 401 session refresh, and preserves multipart
 * uploads, SSE streaming, and AbortController cancellation.
 */

const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  (typeof process !== 'undefined' && process.env && process.env.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8000/api/v1';

let _authConfig = null;

/**
 * Configure centralized session-token resolution and refresh callbacks.
 * Pass null to clear credentials cleanly (e.g. test teardown).
 */
export function configureAuthClient(config) {
  if (!config) {
    _authConfig = null;
    return;
  }
  _authConfig = {
    getAccessToken: config.getAccessToken || null,
    refreshAccessToken: config.refreshAccessToken || null,
    onUnauthorized: config.onUnauthorized || null,
  };
}

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

async function resolveAccessToken() {
  if (!_authConfig || typeof _authConfig.getAccessToken !== 'function') {
    throw new ApiError('Authentication required. Please sign in to continue.', 401, 'missing_session');
  }
  const token = await _authConfig.getAccessToken();
  const cleanToken = typeof token === 'string' ? token.trim() : '';
  if (!cleanToken) {
    throw new ApiError('Authentication required. Please sign in to continue.', 401, 'missing_token');
  }
  return cleanToken;
}

/**
 * Dispatch an authenticated request with centralized Bearer JWT injection
 * and a strict single-attempt 401 refresh + retry flow (never loops).
 */
async function authorizedFetch(url, options = {}) {
  const token = await resolveAccessToken();
  const baseHeaders = { ...(options.headers || {}) };
  // Never explicitly overwrite FormData Content-Type boundary
  if (typeof FormData !== 'undefined' && options.body instanceof FormData) {
    delete baseHeaders['Content-Type'];
    delete baseHeaders['content-type'];
  }

  const execute = (bearerToken) =>
    fetch(url, {
      ...options,
      headers: {
        ...baseHeaders,
        Authorization: `Bearer ${bearerToken}`,
      },
    });

  let response = await execute(token);
  if (response.status !== 401) {
    return response;
  }

  let refreshedToken = '';
  if (_authConfig && typeof _authConfig.refreshAccessToken === 'function') {
    try {
      const candidate = await _authConfig.refreshAccessToken();
      refreshedToken = typeof candidate === 'string' ? candidate.trim() : '';
    } catch {
      refreshedToken = '';
    }
  }

  if (!refreshedToken || refreshedToken === token) {
    if (_authConfig && typeof _authConfig.onUnauthorized === 'function') {
      await _authConfig.onUnauthorized();
    }
    return response;
  }

  // Strict single retry with newly refreshed token; any subsequent 401 fails closed
  response = await execute(refreshedToken);
  if (response.status === 401 && _authConfig && typeof _authConfig.onUnauthorized === 'function') {
    await _authConfig.onUnauthorized();
  }
  return response;
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
    errorDetail = (await response.text()) || response.statusText;
  }

  let userFriendlyMessage = `Server error (${response.status}): ${errorDetail}`;
  if (response.status === 401) {
    userFriendlyMessage = errorDetail
      ? `Authentication failed: ${errorDetail}`
      : 'Your session has expired. Please sign in again.';
  } else if (response.status === 403) {
    userFriendlyMessage = `Access denied: ${errorDetail}`;
  } else if (response.status === 404) {
    userFriendlyMessage = `Resource not found: ${errorDetail}`;
  } else if (response.status === 409) {
    userFriendlyMessage = `Conflict: ${errorDetail}`;
  } else if (response.status === 400) {
    userFriendlyMessage = `Invalid request: ${errorDetail}`;
  } else if (response.status === 413) {
    userFriendlyMessage = `File too large: ${errorDetail}`;
  } else if (response.status === 429) {
    userFriendlyMessage = 'Rate limit exceeded. Please wait a moment before trying again.';
  } else if (response.status === 502 || response.status === 503 || response.status === 504) {
    userFriendlyMessage = `Backend service unavailable: ${errorDetail}`;
  }

  throw new ApiError(userFriendlyMessage, response.status, errorDetail);
}

/**
 * Fetch public health status of the FastAPI backend.
 */
export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Cannot connect to Retrieva AI backend server. Ensure uvicorn is running.', 0);
  }
}

/**
 * Fetch list of owned documents for the authenticated user.
 */
export async function fetchDocuments({ signal = null } = {}) {
  try {
    const res = await authorizedFetch(`${API_BASE_URL}/documents`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError('Failed to fetch document catalog from backend.', 0);
  }
}

/**
 * Upload a PDF or TXT document file under the authenticated user's session.
 */
export async function uploadDocument(file, { signal = null } = {}) {
  if (!file) {
    throw new ApiError('No file provided for upload.', 400);
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await authorizedFetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
      signal,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError(`Network error while uploading '${file.name}': ${err.message}`, 0);
  }
}

/**
 * Delete an owned ingested document by its UUID.
 */
export async function deleteDocument(documentId) {
  if (!documentId) {
    throw new ApiError('Document ID is required for deletion.', 400);
  }

  try {
    const res = await authorizedFetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}`, {
      method: 'DELETE',
      headers: { Accept: 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Failed to delete document '${documentId}': ${err.message}`, 0);
  }
}

/**
 * Ask a question to the RAG pipeline (non-streaming).
 */
export async function askQuestion({
  query,
  retrieval_mode = 'hybrid',
  top_k = 7,
  document_ids = null,
  conversation_id = null,
  signal = null,
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
      top_k: Number(top_k) || 7,
      candidate_k: 20,
      enable_reranking: true,
      similarity_threshold: null,
      rrf_k: 60,
    },
  };

  try {
    const res = await authorizedFetch(`${API_BASE_URL}/answer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError(`Error communicating with RAG answer generation service: ${err.message}`, 0);
  }
}

/**
 * Execute standalone retrieval search against the authenticated user's knowledge base.
 */
export async function searchRetrieval({
  query,
  mode = 'hybrid',
  top_k = 5,
  candidate_k = 20,
  enable_reranking = false,
  similarity_threshold = null,
  document_id_filter = null,
  document_ids_filter = null,
  signal = null,
} = {}) {
  const cleanQuery = (query || '').trim();
  if (!cleanQuery) {
    throw new ApiError('Query string cannot be empty.', 400);
  }

  const cleanMultiFilter =
    Array.isArray(document_ids_filter) && document_ids_filter.length > 0 ? document_ids_filter : null;

  const payload = {
    query: cleanQuery,
    config: {
      mode: (mode || 'hybrid').toLowerCase(),
      top_k: Number(top_k) || 5,
      candidate_k: Math.max(Number(candidate_k) || 20, Number(top_k) || 5),
      enable_reranking: Boolean(enable_reranking),
      similarity_threshold: similarity_threshold !== null ? Number(similarity_threshold) : null,
      document_id_filter: document_id_filter || null,
      document_ids_filter: cleanMultiFilter,
    },
  };

  try {
    const res = await authorizedFetch(`${API_BASE_URL}/retrieval/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError(`Error communicating with retrieval service: ${err.message}`, 0);
  }
}

/**
 * Stream an answer incrementally from the RAG pipeline via Server-Sent Events (SSE).
 */
export async function askQuestionStream({
  query,
  retrieval_mode = 'hybrid',
  top_k = 7,
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
      top_k: Number(top_k) || 7,
      candidate_k: 20,
      enable_reranking: true,
      similarity_threshold: null,
      rrf_k: 60,
    },
  };

  let res;
  try {
    res = await authorizedFetch(`${API_BASE_URL}/answer/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (fetchErr) {
    if (signal?.aborted || fetchErr.name === 'AbortError') {
      return;
    }
    if (fetchErr instanceof ApiError) {
      onError({ code: fetchErr.status === 401 ? 'unauthorized' : 'http_error', message: fetchErr.message });
      throw fetchErr;
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
    onError({ code: res.status === 401 ? 'unauthorized' : 'http_error', message: apiErr.message });
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
 * Create a new persistent conversation owned by the authenticated user.
 */
export async function createConversation(titleOrOptions = null) {
  let title = null;
  if (typeof titleOrOptions === 'string') {
    title = titleOrOptions;
  } else if (titleOrOptions && typeof titleOrOptions === 'object' && titleOrOptions.title) {
    title = titleOrOptions.title;
  }

  try {
    const res = await authorizedFetch(`${API_BASE_URL}/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
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
 * Fetch all persistent conversations owned by the authenticated user.
 */
export async function fetchConversations(limit = 50, offset = 0, { signal = null } = {}) {
  try {
    const res = await authorizedFetch(`${API_BASE_URL}/conversations?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError(`Error fetching conversations: ${err.message}`, 0);
  }
}

/**
 * Retrieve an owned conversation and its chronological message history.
 */
export async function fetchConversation(conversationId, { signal = null } = {}) {
  const cleanId = (conversationId || '').trim();
  if (!cleanId) {
    throw new ApiError('Conversation ID cannot be empty.', 400);
  }
  try {
    const res = await authorizedFetch(`${API_BASE_URL}/conversations/${encodeURIComponent(cleanId)}`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError(`Error fetching conversation '${conversationId}': ${err.message}`, 0);
  }
}

/**
 * Rename an owned conversation.
 */
export async function updateConversationTitle(conversationId, title) {
  const cleanId = (conversationId || '').trim();
  const cleanTitle = (title || '').trim();
  if (!cleanId) {
    throw new ApiError('Conversation ID cannot be empty.', 400);
  }
  if (!cleanTitle) {
    throw new ApiError('Conversation title cannot be empty.', 400);
  }
  try {
    const res = await authorizedFetch(`${API_BASE_URL}/conversations/${encodeURIComponent(cleanId)}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ title: cleanTitle }),
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error renaming conversation '${conversationId}': ${err.message}`, 0);
  }
}

/**
 * Delete an owned conversation and cascade deletion to associated messages.
 */
export async function deleteConversation(conversationId) {
  const cleanId = (conversationId || '').trim();
  if (!cleanId) {
    throw new ApiError('Conversation ID cannot be empty.', 400);
  }
  try {
    const res = await authorizedFetch(`${API_BASE_URL}/conversations/${encodeURIComponent(cleanId)}`, {
      method: 'DELETE',
      headers: { Accept: 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Error deleting conversation '${conversationId}': ${err.message}`, 0);
  }
}
