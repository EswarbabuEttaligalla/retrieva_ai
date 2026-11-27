import { test, describe, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { 
  ApiError, 
  configureAuthClient,
  askQuestion, 
  askQuestionStream, 
  uploadDocument, 
  deleteDocument, 
  searchRetrieval,
  createConversation,
  fetchConversations,
  fetchConversation,
  updateConversationTitle,
  deleteConversation,
} from '../src/api/client.js';

describe('Frontend API Client Suite', () => {
  beforeEach(() => {
    configureAuthClient({
      getAccessToken: async () => 'test-supabase-jwt-token',
    });
  });

  afterEach(() => {
    configureAuthClient(null);
  });
  test('ApiError instantiates correctly with status and detail', () => {
    const err = new ApiError('Not found', 404, 'Document doc_123 does not exist');
    assert.equal(err.name, 'ApiError');
    assert.equal(err.status, 404);
    assert.equal(err.message, 'Not found');
    assert.equal(err.detail, 'Document doc_123 does not exist');
  });

  test('uploadDocument rejects missing file', async () => {
    await assert.rejects(
      async () => {
        await uploadDocument(null);
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'No file provided for upload.',
      }
    );
  });

  test('deleteDocument rejects empty document ID', async () => {
    await assert.rejects(
      async () => {
        await deleteDocument('');
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'Document ID is required for deletion.',
      }
    );
  });

  test('askQuestion rejects empty query string', async () => {
    await assert.rejects(
      async () => {
        await askQuestion({ query: '   ' });
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'Query string cannot be empty.',
      }
    );
  });

  test('searchRetrieval rejects empty query string', async () => {
    await assert.rejects(
      async () => {
        await searchRetrieval({ query: '   ' });
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'Query string cannot be empty.',
      }
    );
  });

  test('searchRetrieval dispatches correctly structured payload to endpoint', async () => {
    const originalFetch = globalThis.fetch;
    let capturedUrl = null;
    let capturedOptions = null;

    globalThis.fetch = async (url, options) => {
      capturedUrl = url;
      capturedOptions = options;
      return {
        ok: true,
        json: async () => ({
          query: 'quantum qubits',
          mode: 'hybrid',
          total_candidates_found: 2,
          returned_chunks_count: 2,
          chunks: [],
          execution_time_ms: 12.5,
          configuration_applied: {},
          reranking_summary: null,
        }),
      };
    };

    try {
      const res = await searchRetrieval({
        query: 'quantum qubits',
        mode: 'hybrid',
        top_k: 3,
        candidate_k: 10,
        enable_reranking: true,
        document_id_filter: 'doc_42',
      });

      assert.ok(capturedUrl.endsWith('/retrieval/search'));
      assert.equal(capturedOptions.method, 'POST');
      const body = JSON.parse(capturedOptions.body);
      assert.equal(body.query, 'quantum qubits');
      assert.equal(body.config.mode, 'hybrid');
      assert.equal(body.config.top_k, 3);
      assert.equal(body.config.candidate_k, 10);
      assert.equal(body.config.enable_reranking, true);
      assert.equal(body.config.document_id_filter, 'doc_42');
      assert.equal(res.returned_chunks_count, 2);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('askQuestionStream rejects empty query string', async () => {
    let errorCalled = false;
    await assert.rejects(
      async () => {
        await askQuestionStream({
          query: '   ',
          onError: (e) => {
            errorCalled = true;
            assert.equal(e.code, 'invalid_request');
          },
        });
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'Query string cannot be empty.',
      }
    );
    assert.equal(errorCalled, true);
  });

  test('askQuestionStream parses SSE events accurately', async () => {
    const originalFetch = globalThis.fetch;

    const ssePayload = [
      'event: status\ndata: {"stage": "retrieving", "message": "Searching..."}\n\n',
      'event: token\ndata: {"delta": "Hello "}\n\n',
      'event: token\ndata: {"delta": "world."}\n\n',
      'event: citations\ndata: {"status": "verified", "citations": [{"source_id": "[Source 1]"}], "metadata": {"latency_ms": 150}}\n\n',
      'event: done\ndata: {"finished": true}\n\n',
    ].join('');

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(ssePayload));
        controller.close();
      },
    });

    globalThis.fetch = async () => ({
      ok: true,
      body: stream,
    });

    const statuses = [];
    const tokens = [];
    let receivedCitations = null;
    let doneCalled = false;

    try {
      await askQuestionStream({
        query: 'test question',
        onStatus: (s) => statuses.push(s),
        onToken: (t) => tokens.push(t),
        onCitations: (c) => {
          receivedCitations = c;
        },
        onDone: (d) => {
          doneCalled = true;
        },
      });

      assert.equal(statuses.length, 1);
      assert.equal(statuses[0].stage, 'retrieving');
      assert.equal(tokens.join(''), 'Hello world.');
      assert.equal(receivedCitations.status, 'verified');
      assert.equal(receivedCitations.citations.length, 1);
      assert.equal(doneCalled, true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('askQuestionStream handles fragmented SSE chunks across chunk boundaries', async () => {
    const originalFetch = globalThis.fetch;

    // Split frames across multiple arbitrary chunks
    const chunk1 = 'event: token\nda';
    const chunk2 = 'ta: {"delta": "Chunked ';
    const chunk3 = 'token"}\n\nevent: done\ndata: {"finished": true}\n\n';

    const encoder = new TextEncoder();
    const chunks = [chunk1, chunk2, chunk3];
    let i = 0;

    const stream = new ReadableStream({
      pull(controller) {
        if (i < chunks.length) {
          controller.enqueue(encoder.encode(chunks[i++]));
        } else {
          controller.close();
        }
      },
    });

    globalThis.fetch = async () => ({
      ok: true,
      body: stream,
    });

    const tokens = [];
    let doneFinished = false;

    try {
      await askQuestionStream({
        query: 'fragment test',
        onToken: (t) => tokens.push(t),
        onDone: (d) => {
          doneFinished = Boolean(d.finished);
        },
      });

      assert.equal(tokens.join(''), 'Chunked token');
      assert.equal(doneFinished, true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('askQuestionStream routes rejection events to onRejection', async () => {
    const originalFetch = globalThis.fetch;

    const ssePayload = [
      'event: token\ndata: {"delta": "Fabricated text [Source 99]."}\n\n',
      'event: rejection\ndata: {"status": "rejected", "reason": "Fabricated citation", "safe_fallback": "Safe fallback text."}\n\n',
      'event: done\ndata: {"finished": true}\n\n',
    ].join('');

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(ssePayload));
        controller.close();
      },
    });

    globalThis.fetch = async () => ({
      ok: true,
      body: stream,
    });

    let rejectionPayload = null;
    let citationsCalled = false;

    try {
      await askQuestionStream({
        query: 'test query',
        onCitations: () => {
          citationsCalled = true;
        },
        onRejection: (rej) => {
          rejectionPayload = rej;
        },
      });

      assert.equal(citationsCalled, false);
      assert.ok(rejectionPayload);
      assert.equal(rejectionPayload.status, 'rejected');
      assert.equal(rejectionPayload.safe_fallback, 'Safe fallback text.');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('askQuestionStream handles AbortSignal cancellation cleanly', async () => {
    const originalFetch = globalThis.fetch;
    const controller = new AbortController();

    globalThis.fetch = async (url, options) => {
      // Check signal
      if (options.signal?.aborted) {
        const err = new Error('The operation was aborted');
        err.name = 'AbortError';
        throw err;
      }
      return {
        ok: true,
        body: new ReadableStream({
          start(c) {
            c.close();
          },
        }),
      };
    };

    controller.abort();

    try {
      // Should exit cleanly without throwing unhandled rejection
      await askQuestionStream({
        query: 'aborted test',
        signal: controller.signal,
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('createConversation dispatches POST request with title', async () => {
    const originalFetch = globalThis.fetch;
    let requestedUrl = '';
    let requestedOptions = null;

    globalThis.fetch = async (url, options) => {
      requestedUrl = url;
      requestedOptions = options;
      return {
        ok: true,
        json: async () => ({
          id: 'conv_123',
          title: 'Quantum Computing',
          created_at: '2026-11-23T12:00:00Z',
          updated_at: '2026-11-23T12:00:00Z',
          message_count: 0,
        }),
      };
    };

    try {
      const res = await createConversation({ title: 'Quantum Computing' });
      assert.ok(requestedUrl.endsWith('/conversations'));
      assert.equal(requestedOptions.method, 'POST');
      assert.deepEqual(JSON.parse(requestedOptions.body), { title: 'Quantum Computing' });
      assert.equal(res.id, 'conv_123');
      assert.equal(res.title, 'Quantum Computing');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('fetchConversations retrieves paginated conversation summaries', async () => {
    const originalFetch = globalThis.fetch;
    let requestedUrl = '';

    globalThis.fetch = async (url) => {
      requestedUrl = url;
      return {
        ok: true,
        json: async () => [
          { id: 'conv_1', title: 'Session 1', message_count: 2 },
          { id: 'conv_2', title: 'Session 2', message_count: 4 },
        ],
      };
    };

    try {
      const res = await fetchConversations(10, 5);
      assert.ok(requestedUrl.includes('/conversations?limit=10&offset=5'));
      assert.equal(res.length, 2);
      assert.equal(res[0].id, 'conv_1');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('fetchConversation rejects empty ID and retrieves conversation detail', async () => {
    await assert.rejects(
      async () => {
        await fetchConversation('');
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'Conversation ID cannot be empty.',
      }
    );

    const originalFetch = globalThis.fetch;
    let requestedUrl = '';

    globalThis.fetch = async (url) => {
      requestedUrl = url;
      return {
        ok: true,
        json: async () => ({
          id: 'conv_abc',
          title: 'Deep Learning',
          messages: [
            { id: 'msg_1', role: 'user', content: 'What is backpropagation?' },
          ],
        }),
      };
    };

    try {
      const res = await fetchConversation('conv_abc');
      assert.ok(requestedUrl.endsWith('/conversations/conv_abc'));
      assert.equal(res.id, 'conv_abc');
      assert.equal(res.messages.length, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('deleteConversation rejects empty ID and issues DELETE call', async () => {
    await assert.rejects(
      async () => {
        await deleteConversation('   ');
      },
      {
        name: 'ApiError',
        status: 400,
        message: 'Conversation ID cannot be empty.',
      }
    );

    const originalFetch = globalThis.fetch;
    let requestedUrl = '';
    let requestedMethod = '';

    globalThis.fetch = async (url, options) => {
      requestedUrl = url;
      requestedMethod = options.method;
      return {
        ok: true,
        json: async () => ({ status: 'success', deleted: true, conversation_id: 'conv_del' }),
      };
    };

    try {
      const res = await deleteConversation('conv_del');
      assert.ok(requestedUrl.endsWith('/conversations/conv_del'));
      assert.equal(requestedMethod, 'DELETE');
      assert.equal(res.deleted, true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('askQuestion and askQuestionStream forward conversation_id in payload', async () => {
    const originalFetch = globalThis.fetch;
    let jsonBody = null;
    let streamBody = null;

    globalThis.fetch = async (url, options) => {
      if (url.endsWith('/answer')) {
        jsonBody = JSON.parse(options.body);
        return {
          ok: true,
          json: async () => ({
            answer: 'Mock answer',
            citations: [],
            sources_used: [],
            conversation_id: 'conv_test',
          }),
        };
      }
      if (url.endsWith('/answer/stream')) {
        streamBody = JSON.parse(options.body);
        return {
          ok: true,
          body: new ReadableStream({
            start(c) {
              c.close();
            },
          }),
        };
      }
      return { ok: false, status: 404 };
    };

    try {
      await askQuestion({
        query: 'Testing conv forwarding',
        conversation_id: 'conv_xyz',
      });
      assert.equal(jsonBody.conversation_id, 'conv_xyz');
      assert.equal(jsonBody.retrieval_config.top_k, 7);
      assert.equal(jsonBody.retrieval_config.similarity_threshold, null);

      await askQuestionStream({
        query: 'Testing conv stream forwarding',
        conversation_id: 'conv_xyz_stream',
      });
      assert.equal(streamBody.conversation_id, 'conv_xyz_stream');
      assert.equal(streamBody.retrieval_config.top_k, 7);
      assert.equal(streamBody.retrieval_config.similarity_threshold, null);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
