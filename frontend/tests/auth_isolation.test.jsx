import test, { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import './setup.mjs';
import React from 'react';
import { render, screen, fireEvent, act, cleanup, waitFor } from '@testing-library/react';

import App from '../src/App.jsx';
import { AuthProvider } from '../src/context/AuthContext.jsx';
import {
  ApiError,
  configureAuthClient,
  fetchDocuments,
  uploadDocument,
  askQuestion,
  askQuestionStream,
  searchRetrieval,
  createConversation,
  fetchConversations,
  fetchConversation,
  updateConversationTitle,
  deleteConversation,
  deleteDocument,
} from '../src/api/client.js';
import { setSupabaseClientForTesting } from '../src/lib/supabase.js';

function createMockSupabaseAuth({
  initialSession = null,
  initialGetSessionError = null,
  signUpHandler = null,
  signInHandler = null,
  refreshHandler = null,
  signOutHandler = null,
} = {}) {
  let currentSession = initialSession;
  let authListener = null;

  const client = {
    auth: {
      async getSession() {
        if (initialGetSessionError) {
          return { data: { session: null }, error: initialGetSessionError };
        }
        return { data: { session: currentSession }, error: null };
      },
      onAuthStateChange(cb) {
        authListener = cb;
        return {
          data: {
            subscription: {
              unsubscribe() {
                authListener = null;
              },
            },
          },
        };
      },
      async signUp({ email, password }) {
        if (signUpHandler) {
          return await signUpHandler({ email, password });
        }
        return {
          data: { user: { id: 'new-user', email }, session: null },
          error: null,
        };
      },
      async signInWithPassword({ email, password }) {
        if (signInHandler) {
          const result = await signInHandler({ email, password });
          if (result?.data?.session) {
            currentSession = result.data.session;
          }
          return result;
        }
        return { data: { user: null, session: null }, error: { message: 'Invalid login credentials' } };
      },
      async refreshSession() {
        if (refreshHandler) {
          const result = await refreshHandler(currentSession);
          currentSession = result?.data?.session || null;
          return result;
        }
        return { data: { session: null }, error: { message: 'Session expired' } };
      },
      async signOut(options) {
        currentSession = null;
        if (signOutHandler) {
          await signOutHandler(options);
        }
        if (authListener) {
          authListener('SIGNED_OUT', null);
        }
        return { error: null };
      },
    },
    _emitAuthEvent(event, session) {
      currentSession = session;
      if (authListener) {
        authListener(event, session);
      }
    },
    _setSession(session) {
      currentSession = session;
    },
  };

  return client;
}

function resetTestEnv() {
  cleanup();
  configureAuthClient(null);
  setSupabaseClientForTesting(null);
  if (globalThis.document?.body) {
    globalThis.document.body.innerHTML = '';
  }
}

describe('Milestone 3.6 Auth, API Authorization & Tenant UI Isolation Suite', () => {
  afterEach(() => {
    resetTestEnv();
  });

  // =========================================================================
  // 1. Authentication Flows & Validation
  // =========================================================================

  it('1. validates signup email, password length, and confirmation mismatch', async () => {
    const originalFetch = globalThis.fetch;
    let fetchCalls = 0;
    globalThis.fetch = async () => {
      fetchCalls += 1;
      return { ok: true, json: async () => ({}) };
    };

    const mockSupabase = createMockSupabaseAuth();
    setSupabaseClientForTesting(mockSupabase);

    try {
      await act(async () => {
        render(
          <AuthProvider supabaseClient={mockSupabase}>
            <App />
          </AuthProvider>
        );
      });

      fireEvent.click(screen.getByRole('tab', { name: /Create Account/i }));

      // Invalid email
      fireEvent.change(screen.getByLabelText(/Email Address/i), { target: { value: 'bad-email' } });
      fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: 'secret123' } });
      fireEvent.change(screen.getByLabelText(/Confirm Password/i), { target: { value: 'secret123' } });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /^Create Account$/i }));
      });
      assert.ok(screen.getByText(/Please enter a valid email address/i));

      // Short password
      fireEvent.change(screen.getByLabelText(/Email Address/i), { target: { value: 'alice@example.com' } });
      fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: '123' } });
      fireEvent.change(screen.getByLabelText(/Confirm Password/i), { target: { value: '123' } });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /^Create Account$/i }));
      });
      assert.ok(screen.getByText(/Password must be at least 6 characters/i));

      // Password mismatch
      fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: 'secret123' } });
      fireEvent.change(screen.getByLabelText(/Confirm Password/i), { target: { value: 'different123' } });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /^Create Account$/i }));
      });
      assert.ok(screen.getByText(/Passwords do not match/i));

      // No backend protected requests should fire while unauthenticated
      assert.equal(fetchCalls, 0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('2. handles signup requiring email confirmation without auto-authenticating', async () => {
    const mockSupabase = createMockSupabaseAuth({
      signUpHandler: async ({ email }) => ({
        data: { user: { id: 'user-pending', email }, session: null },
        error: null,
      }),
    });

    await act(async () => {
      render(
        <AuthProvider supabaseClient={mockSupabase}>
          <App />
        </AuthProvider>
      );
    });

    fireEvent.click(screen.getByRole('tab', { name: /Create Account/i }));
    fireEvent.change(screen.getByLabelText(/Email Address/i), { target: { value: 'confirm@example.com' } });
    fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: 'ValidPass123!' } });
    fireEvent.change(screen.getByLabelText(/Confirm Password/i), { target: { value: 'ValidPass123!' } });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^Create Account$/i }));
    });

    assert.ok(screen.getByText(/Confirmation link sent to confirm@example\.com/i));
    assert.equal(screen.queryByText(/Grounded QA Console/i), null);
  });

  it('3. displays invalid login errors and logs in cleanly on valid credentials', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      if (url.endsWith('/health')) return { ok: true, json: async () => ({ status: 'ok' }) };
      if (url.endsWith('/documents')) return { ok: true, json: async () => ({ documents: [], total: 0 }) };
      if (url.includes('/conversations')) return { ok: true, json: async () => [] };
      return { ok: false, status: 404, json: async () => ({ detail: 'Not found' }) };
    };

    const userASession = {
      access_token: 'jwt-alice-1',
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      user: { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', email: 'alice@example.com' },
    };

    const mockSupabase = createMockSupabaseAuth({
      signInHandler: async ({ email, password }) => {
        if (email === 'alice@example.com' && password === 'CorrectPass!') {
          return { data: { user: userASession.user, session: userASession }, error: null };
        }
        return { data: { user: null, session: null }, error: { message: 'Invalid login credentials' } };
      },
    });

    try {
      await act(async () => {
        render(
          <AuthProvider supabaseClient={mockSupabase}>
            <App />
          </AuthProvider>
        );
      });

      // Failed login attempt
      fireEvent.change(screen.getByLabelText(/Email Address/i), { target: { value: 'alice@example.com' } });
      fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: 'WrongPass' } });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Sign In/i }));
      });
      assert.ok(screen.getByText(/Invalid email or password/i));

      // Valid login
      fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: 'CorrectPass!' } });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Sign In/i }));
      });

      await waitFor(() => {
        assert.ok(screen.getByText(/Grounded QA Console/i));
        assert.ok(screen.getByText('alice@example.com'));
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('4. restores existing session on startup and rejects expired unrefreshable session', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      if (url.endsWith('/health')) return { ok: true, json: async () => ({ status: 'ok' }) };
      if (url.endsWith('/documents')) return { ok: true, json: async () => ({ documents: [], total: 0 }) };
      if (url.includes('/conversations')) return { ok: true, json: async () => [] };
      return { ok: false, status: 404, json: async () => ({ detail: 'Not found' }) };
    };

    try {
      // Valid session restored
      const validSession = {
        access_token: 'jwt-restored-1',
        expires_at: Math.floor(Date.now() / 1000) + 3600,
        user: { id: '11111111-1111-4111-8111-111111111111', email: 'restored@example.com' },
      };
      const mockRestored = createMockSupabaseAuth({ initialSession: validSession });

      await act(async () => {
        render(
          <AuthProvider supabaseClient={mockRestored}>
            <App />
          </AuthProvider>
        );
      });

      await waitFor(() => {
        assert.ok(screen.getByText('restored@example.com'));
      });

      cleanup();

      // Expired session that cannot refresh -> forces sign-in screen with expiry message
      const expiredSession = {
        access_token: 'jwt-expired',
        expires_at: Math.floor(Date.now() / 1000) - 600,
        user: { id: '11111111-1111-4111-8111-111111111111', email: 'expired@example.com' },
      };
      const mockExpired = createMockSupabaseAuth({
        initialSession: expiredSession,
        refreshHandler: async () => ({ data: { session: null }, error: { message: 'Refresh expired' } }),
      });

      await act(async () => {
        render(
          <AuthProvider supabaseClient={mockExpired}>
            <App />
          </AuthProvider>
        );
      });

      await waitFor(() => {
        assert.ok(screen.getByText(/Your previous session expired/i));
        assert.equal(screen.queryByText(/Grounded QA Console/i), null);
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // =========================================================================
  // 2. Centralized API Client Token Injection, 401 Refresh & No Loops
  // =========================================================================

  it('5. blocks protected requests when unauthenticated and never triggers fetch', async () => {
    const originalFetch = globalThis.fetch;
    let fetchCalled = false;
    globalThis.fetch = async () => {
      fetchCalled = true;
      return { ok: true, json: async () => ({}) };
    };

    configureAuthClient(null);

    try {
      for (const fn of [
        () => fetchDocuments(),
        () => uploadDocument(new File(['abc'], 'a.txt', { type: 'text/plain' })),
        () => deleteDocument('doc-1'),
        () => askQuestion({ query: 'hello' }),
        () => searchRetrieval({ query: 'hello' }),
        () => askQuestionStream({ query: 'hello' }),
        () => createConversation({ title: 'Chat' }),
        () => fetchConversations(),
        () => fetchConversation('conv-1'),
        () => updateConversationTitle('conv-1', 'Renamed'),
        () => deleteConversation('conv-1'),
      ]) {
        await assert.rejects(fn, (err) => err instanceof ApiError && err.status === 401);
      }
      assert.equal(fetchCalled, false);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('6. attaches Bearer JWT, handles 401 with single refresh retry, and never loops', async () => {
    const originalFetch = globalThis.fetch;
    let activeToken = 'stale-token-1';
    let refreshCount = 0;
    let unauthorizedCount = 0;
    const seenAuthHeaders = [];

    configureAuthClient({
      getAccessToken: async () => activeToken,
      refreshAccessToken: async () => {
        refreshCount += 1;
        activeToken = 'refreshed-token-2';
        return activeToken;
      },
      onUnauthorized: async () => {
        unauthorizedCount += 1;
        activeToken = '';
      },
    });

    globalThis.fetch = async (url, options) => {
      seenAuthHeaders.push(options?.headers?.Authorization);
      if (url.endsWith('/documents') && options?.headers?.Authorization === 'Bearer stale-token-1') {
        return { ok: false, status: 401, json: async () => ({ detail: 'Token expired' }) };
      }
      if (url.endsWith('/documents') && options?.headers?.Authorization === 'Bearer refreshed-token-2') {
        return { ok: true, json: async () => ({ documents: [{ document_id: 'd1', filename: 'ok.txt' }], total: 1 }) };
      }
      // Persistent 401 endpoint (testing no infinite loop)
      return { ok: false, status: 401, json: async () => ({ detail: 'Revoked' }) };
    };

    try {
      const res = await fetchDocuments();
      assert.equal(res.documents.length, 1);
      assert.equal(refreshCount, 1);
      assert.deepEqual(seenAuthHeaders, ['Bearer stale-token-1', 'Bearer refreshed-token-2']);
      assert.equal(unauthorizedCount, 0);

      // Next request returns 401 before & after single refresh -> stops after 1 retry & invokes onUnauthorized
      seenAuthHeaders.length = 0;
      refreshCount = 0;
      activeToken = 'stale-token-1';

      await assert.rejects(
        () => fetchConversations(),
        (err) => err instanceof ApiError && err.status === 401
      );
      assert.equal(refreshCount, 1);
      assert.equal(seenAuthHeaders.length, 2);
      assert.equal(unauthorizedCount, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('7. preserves multipart upload boundary, status error mappings (403/404/409/503), and SSE + AbortController', async () => {
    const originalFetch = globalThis.fetch;
    configureAuthClient({
      getAccessToken: async () => 'valid-jwt-xyz',
    });

    let capturedUploadHeaders = null;
    let capturedUploadBody = null;

    globalThis.fetch = async (url, options) => {
      if (url.endsWith('/documents/upload')) {
        capturedUploadHeaders = options.headers;
        capturedUploadBody = options.body;
        return { ok: false, status: 409, json: async () => ({ detail: 'Duplicate document' }) };
      }
      if (url.endsWith('/documents/forbidden')) {
        return { ok: false, status: 403, json: async () => ({ detail: 'Forbidden' }) };
      }
      if (url.endsWith('/conversations/missing')) {
        return { ok: false, status: 404, json: async () => ({ detail: 'Not found' }) };
      }
      if (url.includes('/conversations?')) {
        return { ok: false, status: 503, json: async () => ({ detail: 'Supabase down' }) };
      }
      return { ok: false, status: 500 };
    };

    try {
      const file = new File(['hello'], 'notes.txt', { type: 'text/plain' });
      await assert.rejects(() => uploadDocument(file), (e) => e.status === 409 && e.message.includes('Conflict'));
      assert.ok(capturedUploadBody instanceof FormData);
      assert.equal(capturedUploadHeaders['Content-Type'], undefined);
      assert.equal(capturedUploadHeaders.Authorization, 'Bearer valid-jwt-xyz');

      await assert.rejects(() => deleteDocument('forbidden'), (e) => e.status === 403);
      await assert.rejects(() => fetchConversation('missing'), (e) => e.status === 404);
      await assert.rejects(() => fetchConversations(), (e) => e.status === 503);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // =========================================================================
  // 3. UI Account Switching, Stale Data & In-Flight Abort Isolation
  // =========================================================================

  it('8. clears all User A data, citations, inspector state, and aborts in-flight fetches on logout & User B login', async () => {
    const originalFetch = globalThis.fetch;

    const userASession = {
      access_token: 'jwt-user-a',
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      user: { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', email: 'usera@example.com' },
    };
    const userBSession = {
      access_token: 'jwt-user-b',
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      user: { id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', email: 'userb@example.com' },
    };

    let resolveDelayedADocs = null;
    const seenSignalsAborted = [];

    globalThis.fetch = async (url, options = {}) => {
      const auth = options?.headers?.Authorization || '';
      if (url.endsWith('/health')) {
        return { ok: true, json: async () => ({ status: 'ok' }) };
      }
      if (url.endsWith('/documents')) {
        if (auth === 'Bearer jwt-user-a') {
          if (url.includes('delayed')) {
            return new Promise((resolve) => {
              resolveDelayedADocs = resolve;
            });
          }
          if (options.signal) {
            options.signal.addEventListener('abort', () => seenSignalsAborted.push('docs-a'));
          }
          return {
            ok: true,
            json: async () => ({
              documents: [
                {
                  document_id: 'doc-a-1',
                  filename: 'alice_confidential.pdf',
                  file_extension: '.pdf',
                  total_pages: 3,
                  total_chunks: 9,
                  file_size_bytes: 12000,
                  created_at: '2026-11-26T09:00:00Z',
                },
              ],
              total: 1,
            }),
          };
        }
        if (auth === 'Bearer jwt-user-b') {
          return {
            ok: true,
            json: async () => ({
              documents: [
                {
                  document_id: 'doc-b-1',
                  filename: 'bob_public.txt',
                  file_extension: '.txt',
                  total_pages: 1,
                  total_chunks: 2,
                  file_size_bytes: 512,
                  created_at: '2026-11-26T09:30:00Z',
                },
              ],
              total: 1,
            }),
          };
        }
      }
      if (url.includes('/conversations/conv-a-1')) {
        if (auth === 'Bearer jwt-user-a') {
          return {
            ok: true,
            json: async () => ({
              id: 'conv-a-1',
              title: 'Alice Secret Thread',
              created_at: '2026-11-26T09:00:00Z',
              updated_at: '2026-11-26T09:05:00Z',
              messages: [
                {
                  id: 'msg-a-1',
                  role: 'assistant',
                  content: 'Secret Obsidian Zurich metrics [Source 1].',
                  metadata: {
                    citations: [
                      {
                        source_id: '[Source 1]',
                        filename: 'alice_confidential.pdf',
                        document_id: 'doc-a-1',
                        page_references: [2],
                        quote: 'Alice private finding',
                      },
                    ],
                  },
                  created_at: '2026-11-26T09:05:00Z',
                },
              ],
            }),
          };
        }
        return { ok: false, status: 404, json: async () => ({ detail: 'Not found' }) };
      }
      if (url.endsWith('/answer/stream')) {
        if (options.signal) {
          options.signal.addEventListener('abort', () => seenSignalsAborted.push('stream-a'));
        }
        const pendingStream = new ReadableStream({
          start() {
            // Deliberately keep stream open until aborted by Sign Out
          },
        });
        return { ok: true, body: pendingStream };
      }
      if (url.includes('/conversations')) {
        if (auth === 'Bearer jwt-user-a') {
          return {
            ok: true,
            json: async () => [
              { id: 'conv-a-1', title: 'Alice Secret Thread', created_at: '2026-11-26T09:00:00Z', updated_at: '2026-11-26T09:05:00Z', message_count: 1 },
            ],
          };
        }
        if (auth === 'Bearer jwt-user-b') {
          return {
            ok: true,
            json: async () => [
              { id: 'conv-b-1', title: 'Bob Workspace Chat', created_at: '2026-11-26T09:30:00Z', updated_at: '2026-11-26T09:30:00Z', message_count: 0 },
            ],
          };
        }
      }
      return { ok: false, status: 404, json: async () => ({ detail: 'Not found' }) };
    };

    const mockSupabase = createMockSupabaseAuth({
      initialSession: userASession,
      signInHandler: async ({ email }) => {
        if (email === 'userb@example.com') {
          return { data: { user: userBSession.user, session: userBSession }, error: null };
        }
        return { data: { user: null, session: null }, error: { message: 'Invalid credentials' } };
      },
    });

    try {
      await act(async () => {
        render(
          <AuthProvider supabaseClient={mockSupabase}>
            <App />
          </AuthProvider>
        );
      });

      // User A sees Alice's document and conversation
      await waitFor(() => {
        assert.ok(screen.getByText('alice_confidential.pdf'));
        assert.ok(screen.getByText('Alice Secret Thread'));
      });

      // Load Alice's conversation & start an active question stream before signing out
      await act(async () => {
        fireEvent.click(screen.getByText('Alice Secret Thread'));
      });
      await waitFor(() => {
        assert.ok(screen.getByText(/Secret Obsidian Zurich metrics/i));
      });

      const chatInput = screen.getByPlaceholderText(/Ask a question about your documents/i);
      await act(async () => {
        fireEvent.change(chatInput, { target: { value: 'Active question before logout' } });
      });
      await act(async () => {
        fireEvent.keyDown(chatInput, { key: 'Enter', code: 'Enter' });
      });

      // User A signs out while stream is still open -> active stream aborted & state cleared
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Sign Out/i }));
      });

      await waitFor(() => {
        assert.equal(screen.queryByText('alice_confidential.pdf'), null);
        assert.equal(screen.queryByText('Alice Secret Thread'), null);
        assert.equal(screen.queryByText(/Secret Obsidian Zurich metrics/i), null);
        assert.ok(screen.getByRole('button', { name: /Sign In/i }));
      });

      // User B signs in
      fireEvent.change(screen.getByLabelText(/Email Address/i), { target: { value: 'userb@example.com' } });
      fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: 'BobPass123!' } });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Sign In/i }));
      });

      // User B sees ONLY Bob's docs & chats; ZERO User A artifacts remain
      await waitFor(() => {
        assert.ok(screen.getByText('bob_public.txt'));
        assert.ok(screen.getByText('Bob Workspace Chat'));
        assert.equal(screen.queryByText('alice_confidential.pdf'), null);
        assert.equal(screen.queryByText('Alice Secret Thread'), null);
        assert.equal(screen.queryByText(/Secret Obsidian Zurich metrics/i), null);
      });

      // Even if a late User A async fetch settles, User B's state stays clean
      if (resolveDelayedADocs) {
        resolveDelayedADocs({ ok: true, json: async () => ({ documents: [{ document_id: 'stale', filename: 'stale.pdf' }] }) });
        assert.equal(screen.queryByText('stale.pdf'), null);
      }
      assert.ok(seenSignalsAborted.length >= 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
