import test, { describe, it } from 'node:test';
import assert from 'node:assert';
import './setup.mjs';
import React from 'react';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';

import Sidebar from '../src/components/Sidebar.jsx';
import ChatWorkspace from '../src/components/ChatWorkspace.jsx';
import CitationDrawer from '../src/components/CitationDrawer.jsx';
import MarkdownRenderer from '../src/components/MarkdownRenderer.jsx';
import RetrievalInspector from '../src/components/RetrievalInspector.jsx';

function resetDom() {
  cleanup();
  if (globalThis.document && globalThis.document.body) {
    globalThis.document.body.innerHTML = '';
  }
}

describe('Frontend Component Verification Suite', () => {

  // 1. Document list rendering
  it('1. renders document list with filenames, chunk counts, and page indicators', () => {
    resetDom();
    const mockDocs = [
      {
        document_id: 'doc_1',
        filename: 'neural_networks.pdf',
        file_extension: '.pdf',
        file_type: 'pdf',
        total_pages: 5,
        total_chunks: 12,
        file_size_bytes: 204800,
        created_at: '2026-11-23T10:00:00Z',
      },
      {
        document_id: 'doc_2',
        filename: 'thermodynamics.txt',
        file_extension: '.txt',
        file_type: 'txt',
        total_pages: 1,
        total_chunks: 4,
        file_size_bytes: 4096,
        created_at: '2026-11-23T10:30:00Z',
      },
    ];

    render(
      <Sidebar
        documents={mockDocs}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={() => {}}
        onDelete={() => {}}
        onRefresh={() => {}}
      />
    );

    assert.ok(screen.getByText('neural_networks.pdf'));
    assert.ok(screen.getByText('thermodynamics.txt'));
    assert.ok(screen.getByText('12 chunks'));
    assert.ok(screen.getByText('4 chunks'));
    assert.ok(screen.getByText('5 pages'));
  });

  // 2. Empty document-list state
  it('2. renders empty document list placeholder when no documents exist', () => {
    resetDom();
    render(
      <Sidebar
        documents={[]}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={() => {}}
        onDelete={() => {}}
        onRefresh={() => {}}
      />
    );

    assert.ok(screen.getByText('No documents indexed yet'));
    assert.ok(screen.getByText(/Upload a PDF or TXT document above/));
  });

  // 3. Upload success state
  it('3. displays success notification when document upload succeeds', async () => {
    resetDom();
    let uploadCalled = false;
    const mockUpload = async (file) => {
      uploadCalled = true;
      return { status: 'success', total_chunks: 7, document_id: 'doc_new' };
    };

    const { container } = render(
      <Sidebar
        documents={[]}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={mockUpload}
        onDelete={() => {}}
        onRefresh={() => {}}
      />
    );

    const fileInput = container.querySelector('input[type="file"]');
    assert.ok(fileInput, 'File input element must exist');

    const testFile = new File(['Sample document content'], 'sample_upload.pdf', {
      type: 'application/pdf',
    });

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [testFile] } });
    });

    assert.strictEqual(uploadCalled, true);
    assert.ok(screen.getByText("Indexed 'sample_upload.pdf' (7 chunks)."));
  });

  // 4. Upload failure state
  it('4. displays error banner on upload failure or invalid extension', async () => {
    resetDom();
    // 4a. Client-side extension validation rejection
    const { container } = render(
      <Sidebar
        documents={[]}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={async () => {}}
        onDelete={() => {}}
        onRefresh={() => {}}
      />
    );

    const fileInput = container.querySelector('input[type="file"]');
    const invalidFile = new File(['malicious binary'], 'script.exe', {
      type: 'application/x-msdownload',
    });

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [invalidFile] } });
    });

    assert.ok(screen.getByText(/Unsupported file format '\.exe'/));

    // 4b. Server-side upload failure rejection
    resetDom();

    const mockFailingUpload = async () => {
      throw new Error('Connection timeout during embedding computation');
    };

    const { container: container2 } = render(
      <Sidebar
        documents={[]}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={mockFailingUpload}
        onDelete={() => {}}
        onRefresh={() => {}}
      />
    );

    const fileInput2 = container2.querySelector('input[type="file"]');
    const validFile = new File(['valid content'], 'doc.pdf', { type: 'application/pdf' });

    await act(async () => {
      fireEvent.change(fileInput2, { target: { files: [validFile] } });
    });

    assert.ok(screen.getByText('Connection timeout during embedding computation'));
  });

  // 5. Duplicate upload feedback
  it('5. renders duplicate document notification when file was previously indexed', async () => {
    resetDom();
    const mockUploadDuplicate = async (file) => {
      return { status: 'duplicate', total_chunks: 5, document_id: 'doc_existing' };
    };

    const { container } = render(
      <Sidebar
        documents={[]}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={mockUploadDuplicate}
        onDelete={() => {}}
        onRefresh={() => {}}
      />
    );

    const fileInput = container.querySelector('input[type="file"]');
    const duplicateFile = new File(['content'], 'existing_guide.pdf', {
      type: 'application/pdf',
    });

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [duplicateFile] } });
    });

    assert.ok(
      screen.getByText("Document 'existing_guide.pdf' is already indexed in the vector store.")
    );
  });

  // 6. Document deletion interaction
  it('6. opens deletion confirmation prompt and executes onDelete when confirmed', async () => {
    resetDom();
    let deletedId = null;
    const mockDelete = async (id) => {
      deletedId = id;
    };

    const mockDocs = [
      {
        document_id: 'doc_to_delete',
        filename: 'obsolete.pdf',
        file_extension: '.pdf',
        file_type: 'pdf',
        total_pages: 2,
        total_chunks: 3,
        file_size_bytes: 1024,
      },
    ];

    const { container } = render(
      <Sidebar
        documents={mockDocs}
        loading={false}
        uploading={false}
        backendLive={true}
        onUpload={() => {}}
        onDelete={mockDelete}
        onRefresh={() => {}}
      />
    );

    const deleteBtn = container.querySelector('button[title="Delete document"]');
    assert.ok(deleteBtn, 'Delete button must be rendered');

    await act(async () => {
      fireEvent.click(deleteBtn);
    });

    assert.ok(screen.getByText('Delete this document & its vector chunks?'));
    const confirmBtn = screen.getByText('Yes, Delete');

    await act(async () => {
      fireEvent.click(confirmBtn);
    });

    assert.strictEqual(deletedId, 'doc_to_delete');
  });

  // 7. Chat question submission
  it('7. captures question submission and clears input dock', async () => {
    resetDom();
    let submittedQuery = null;
    const mockSubmit = (query) => {
      submittedQuery = query;
    };

    const { container } = render(
      <ChatWorkspace
        messages={[]}
        loading={false}
        documentCount={3}
        onSubmitQuestion={mockSubmit}
      />
    );

    const textarea = container.querySelector('textarea');
    assert.ok(textarea, 'Chat input textarea must exist');

    await act(async () => {
      fireEvent.change(textarea, { target: { value: 'How does vector quantization work?' } });
    });

    const sendBtn = container.querySelector('button[title="Send question"]');
    assert.ok(sendBtn, 'Send button must exist');

    await act(async () => {
      fireEvent.click(sendBtn);
    });

    assert.strictEqual(submittedQuery, 'How does vector quantization work?');
    assert.strictEqual(textarea.value, '');
  });

  // 8. Loading state during answer generation
  it('8. renders spinning loading indicator and message during active generation', () => {
    resetDom();
    const { getByText } = render(
      <ChatWorkspace
        messages={[{ role: 'user', content: 'What is RRF?' }]}
        loading={true}
        documentCount={2}
      />
    );

    assert.ok(
      getByText(/Searching knowledge base & synthesizing answer/)
    );
  });

  // 9. Answer rendering
  it('9. renders markdown answer with structured elements and metadata', () => {
    resetDom();
    const mockMessages = [
      {
        role: 'user',
        content: 'Explain RAG benefits.',
      },
      {
        role: 'assistant',
        content: 'RAG provides **factual grounding** and `provenance`.\n- Reduces hallucinations\n- Real-time updates',
        citations: [],
        metadata: {
          retrieval_mode: 'hybrid',
          latency_ms: 142.5,
          context_chunks_used: 2,
          total_chunks_retrieved: 5,
        },
      },
    ];

    const { container } = render(
      <ChatWorkspace
        messages={mockMessages}
        loading={false}
        documentCount={2}
      />
    );

    const strongEl = container.querySelector('strong');
    assert.ok(strongEl);
    assert.strictEqual(strongEl.textContent, 'factual grounding');

    const codeEl = container.querySelector('code');
    assert.ok(codeEl);
    assert.strictEqual(codeEl.textContent, 'provenance');

    assert.ok(screen.getByText('Reduces hallucinations'));
    assert.ok(screen.getByText('Real-time updates'));

    assert.ok(screen.getByText('142.5 ms'));
    assert.ok(screen.getByText('2 / 5 chunks used'));
  });

  // 10. Citation button interaction and source drawer
  it('10. renders clickable citation buttons and opens citation drawer with provenance', async () => {
    resetDom();
    let clickedCitation = null;
    const mockCitation = {
      source_id: 'Source 1',
      chunk_id: 'chunk_abc123',
      document_id: 'doc_xyz',
      filename: 'quantum_algorithms.pdf',
      page_references: [3],
      score: 0.884,
      rank: 1,
      quote: 'Shor algorithm achieves polynomial time prime factorization.',
    };

    const mockMessages = [
      {
        role: 'assistant',
        content: 'Quantum computers factor numbers efficiently [Source 1].',
        citations: [mockCitation],
      },
    ];

    render(
      <ChatWorkspace
        messages={mockMessages}
        loading={false}
        documentCount={1}
        onCitationClick={(cit) => {
          clickedCitation = cit;
        }}
      />
    );

    const citationBtn = screen.getByTitle('Click to inspect Source 1');
    assert.ok(citationBtn);

    await act(async () => {
      fireEvent.click(citationBtn);
    });

    assert.ok(clickedCitation);
    assert.strictEqual(clickedCitation.filename, 'quantum_algorithms.pdf');

    resetDom();

    const { container: drawerContainer } = render(
      <CitationDrawer
        isOpen={true}
        citation={mockCitation}
        onClose={() => {}}
      />
    );

    const heading = drawerContainer.querySelector('h3');
    assert.ok(heading && heading.textContent.includes('Source 1'));
    const filenameMatches = screen.getAllByText('quantum_algorithms.pdf');
    assert.ok(filenameMatches.length >= 1);
    assert.ok(screen.getByText(/Shor algorithm achieves polynomial time/));
    assert.ok(screen.getByText('chunk_abc123'));
    assert.ok(screen.getByText('0.8840'));
  });

  // 11. Insufficient-context rendering
  it('11. displays insufficient-context warning alert when knowledge base lacks data', () => {
    resetDom();
    const mockMessages = [
      {
        role: 'assistant',
        content: 'I could not find sufficient information in the indexed documents.',
        insufficient_context: true,
        citations: [],
      },
    ];

    render(
      <ChatWorkspace
        messages={mockMessages}
        loading={false}
        documentCount={1}
      />
    );

    assert.ok(screen.getByText(/Insufficient Document Context:/));
    assert.ok(
      screen.getByText(
        /The ingested knowledge base did not contain enough substantiated information/
      )
    );
  });

  // 12. Retrieval mode selection
  it('12. allows selecting Dense, BM25, and Hybrid retrieval modes', async () => {
    resetDom();
    let selectedMode = null;
    const mockSelect = (mode) => {
      selectedMode = mode;
    };

    const { container } = render(
      <ChatWorkspace
        messages={[]}
        loading={false}
        retrievalMode="hybrid"
        onSelectRetrievalMode={mockSelect}
      />
    );

    const denseBtn = container.querySelector('button[title*="Dense Semantic Vector Search"]');
    const bm25Btn = container.querySelector('button[title*="BM25 Keyword Frequency Search"]');
    const hybridBtn = container.querySelector('button[title*="Hybrid Fusion"]');

    assert.ok(denseBtn, 'Dense mode button must exist');
    assert.ok(bm25Btn, 'BM25 mode button must exist');
    assert.ok(hybridBtn, 'Hybrid mode button must exist');

    await act(async () => {
      fireEvent.click(denseBtn);
    });
    assert.strictEqual(selectedMode, 'dense');

    await act(async () => {
      fireEvent.click(bm25Btn);
    });
    assert.strictEqual(selectedMode, 'bm25');

    await act(async () => {
      fireEvent.click(hybridBtn);
    });
    assert.strictEqual(selectedMode, 'hybrid');
  });

  // 13. XSS and Malicious HTML Sanitization
  it('13. prevents XSS and never injects unsanitized raw HTML or script tags', () => {
    resetDom();
    const maliciousPayload =
      "<script>window.__xss_executed = true;</script><img src=x onerror=alert(1)> [Evil Link](javascript:alert(1))";

    const { container } = render(
      <MarkdownRenderer content={maliciousPayload} />
    );

    const scriptElements = container.querySelectorAll('script');
    assert.strictEqual(scriptElements.length, 0, 'No script tags should exist');

    const imgElements = container.querySelectorAll('img');
    assert.strictEqual(imgElements.length, 0, 'No img tags should exist');

    const anchorElements = container.querySelectorAll('a');
    for (const a of anchorElements) {
      assert.ok(!a.href.startsWith('javascript:'), 'javascript: hrefs are prohibited');
    }

    assert.strictEqual(globalThis.window.__xss_executed, undefined);
  });

  // 14. Retrieval Inspector: renders controls, inputs, mode pills and sliders
  it('14. renders Retrieval Inspector with strategy modes, sliders, and controls', () => {
    resetDom();
    const mockDocs = [
      { document_id: 'doc_1', filename: 'ai_overview.pdf' },
      { document_id: 'doc_2', filename: 'quantum.txt' },
    ];

    render(
      <RetrievalInspector
        isOpen={true}
        onClose={() => {}}
        documents={mockDocs}
      />
    );

    assert.ok(screen.getByText('Retrieval Inspector'));
    assert.ok(screen.getByText('DIAGNOSTIC'));
    assert.ok(screen.getByPlaceholderText(/Enter search query to inspect retrieval candidates/i));
    assert.ok(screen.getByText('DENSE'));
    assert.ok(screen.getByText('BM25'));
    assert.ok(screen.getByText('HYBRID'));
    assert.ok(screen.getByText('Top-K Return'));
    assert.ok(screen.getByText('Candidate Pool'));
    assert.ok(screen.getByLabelText(/Enable Cross-Encoder Reranking/i));
    assert.ok(screen.getByText('All Documents'));
    assert.ok(screen.getByText('ai_overview.pdf'));
    assert.ok(screen.getByText('quantum.txt'));
  });

  // 15. Retrieval Inspector: executes search and displays candidate chunks with score badges
  it('15. executes retrieval search and renders chunk cards with multi-stage score badges', async () => {
    resetDom();
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: true,
      json: async () => ({
        query: 'quantum superposition',
        mode: 'hybrid',
        total_candidates_found: 2,
        returned_chunks_count: 1,
        execution_time_ms: 18.42,
        chunks: [
          {
            chunk_id: 'chunk_alpha_01',
            document_id: 'doc_alpha',
            filename: 'quantum_physics.txt',
            page_number: 1,
            page_references: [1],
            chunk_text: 'Quantum computing leverages superposition and quantum entanglement.',
            start_offset: 0,
            end_offset: 68,
            rank: 1,
            cosine_similarity: 0.8951,
            cosine_distance: 0.1049,
            bm25_score: 3.421,
            rrf_score: 0.0328,
            cross_encoder_score: 4.1205,
            metadata: {},
          },
        ],
        configuration_applied: {},
        reranking_summary: {
          model_name: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
          candidates_in: 2,
          candidates_out: 1,
          duration_ms: 8.5,
        },
      }),
    });

    try {
      render(
        <RetrievalInspector
          isOpen={true}
          onClose={() => {}}
          documents={[{ document_id: 'doc_alpha', filename: 'quantum_physics.txt' }]}
        />
      );

      const input = screen.getByPlaceholderText(/Enter search query to inspect retrieval candidates/i);
      await act(async () => {
        fireEvent.change(input, { target: { value: 'quantum superposition' } });
      });

      const searchBtn = screen.getByRole('button', { name: /search/i });
      await act(async () => {
        fireEvent.click(searchBtn);
      });

      // Verify diagnostic metrics
      assert.ok(screen.getByText('18.42 ms'));
      assert.ok(screen.getByText('2 found'));
      assert.ok(screen.getByText('1 chunks'));
      assert.ok(screen.getByText(/Reranked 2 → 1/i));

      // Verify chunk card
      assert.ok(screen.getByText('#1'));
      assert.ok(screen.getAllByText('quantum_physics.txt').length >= 1);
      assert.ok(screen.getByText('Page 1'));
      assert.ok(screen.getByText('Offsets 0–68'));
      assert.ok(screen.getByText(/Quantum computing leverages superposition/i));

      // Verify score badges
      assert.ok(screen.getByText('0.8951')); // Cosine Sim
      assert.ok(screen.getByText('0.1049')); // Cosine Distance
      assert.ok(screen.getByText('3.4210')); // BM25
      assert.ok(screen.getByText('0.0328')); // RRF
      assert.ok(screen.getByText('4.1205')); // Cross-Encoder
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 16. Retrieval Inspector: displays empty results state
  it('16. renders empty results placeholder when search returns zero candidates', async () => {
    resetDom();
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: true,
      json: async () => ({
        query: 'unknown topic',
        mode: 'dense',
        total_candidates_found: 0,
        returned_chunks_count: 0,
        execution_time_ms: 4.2,
        chunks: [],
        configuration_applied: {},
        reranking_summary: null,
      }),
    });

    try {
      render(
        <RetrievalInspector
          isOpen={true}
          onClose={() => {}}
          documents={[]}
        />
      );

      const input = screen.getByPlaceholderText(/Enter search query to inspect retrieval candidates/i);
      await act(async () => {
        fireEvent.change(input, { target: { value: 'unknown topic' } });
      });

      const searchBtn = screen.getByRole('button', { name: /search/i });
      await act(async () => {
        fireEvent.click(searchBtn);
      });

      assert.ok(screen.getByText('No Candidate Chunks Found'));
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 17. Retrieval Inspector: displays error alert on failure
  it('17. renders error banner when retrieval search throws an error', async () => {
    resetDom();
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: false,
      status: 500,
      json: async () => ({
        detail: 'ChromaDB vector store connection failed',
      }),
    });

    try {
      render(
        <RetrievalInspector
          isOpen={true}
          onClose={() => {}}
          documents={[]}
        />
      );

      const input = screen.getByPlaceholderText(/Enter search query to inspect retrieval candidates/i);
      await act(async () => {
        fireEvent.change(input, { target: { value: 'test fault' } });
      });

      const searchBtn = screen.getByRole('button', { name: /search/i });
      await act(async () => {
        fireEvent.click(searchBtn);
      });

      assert.ok(screen.getByText('Retrieval Search Error'));
      assert.ok(screen.getByText(/ChromaDB vector store connection failed/i));
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 18. ChatWorkspace: triggers onOpenRetrievalInspector from header button
  it('18. triggers onOpenRetrievalInspector when header button is clicked', async () => {
    resetDom();
    let inspectorOpened = false;

    render(
      <ChatWorkspace
        messages={[]}
        loading={false}
        error={null}
        retrievalMode="hybrid"
        onSelectRetrievalMode={() => {}}
        onSubmitQuestion={() => {}}
        onClearChat={() => {}}
        onCitationClick={() => {}}
        onOpenRetrievalInspector={() => { inspectorOpened = true; }}
        documentCount={3}
      />
    );

    const inspectBtn = screen.getByRole('button', { name: /inspect retrieval/i });
    assert.ok(inspectBtn);

    await act(async () => {
      fireEvent.click(inspectBtn);
    });

    assert.strictEqual(inspectorOpened, true);
  });

  // 19. ChatWorkspace: renders in-flight streaming message with provisional status badge
  it('19. renders in-flight streaming message with provisional status badge', () => {
    resetDom();
    const streamingMessages = [
      {
        id: 'msg_1',
        role: 'user',
        content: 'Explain second law',
      },
      {
        id: 'msg_2',
        role: 'assistant',
        content: 'Entropy always increases',
        isStreaming: true,
        isVerified: false,
        statusStage: 'generating',
        statusMessage: 'Streaming grounded response...',
        citations: [],
      },
    ];

    render(
      <ChatWorkspace
        messages={streamingMessages}
        loading={true}
        retrievalMode="hybrid"
        onSelectRetrievalMode={() => {}}
        onSubmitQuestion={() => {}}
        onStopGeneration={() => {}}
        documentCount={1}
      />
    );

    assert.ok(screen.getByText('Provisional'));
    assert.ok(screen.getByText('Streaming grounded response...'));
    assert.ok(screen.getByText(/Entropy always increases/));
  });

  // 20. ChatWorkspace: renders Stop Generation button and fires onStopGeneration
  it('20. renders Stop Generation button during active generation and executes onStopGeneration', async () => {
    resetDom();
    let stopTriggered = false;

    render(
      <ChatWorkspace
        messages={[]}
        loading={true}
        retrievalMode="hybrid"
        onSelectRetrievalMode={() => {}}
        onSubmitQuestion={() => {}}
        onStopGeneration={() => {
          stopTriggered = true;
        }}
        documentCount={1}
      />
    );

    const stopBtn = screen.getByRole('button', { name: /stop/i });
    assert.ok(stopBtn);

    await act(async () => {
      fireEvent.click(stopBtn);
    });

    assert.strictEqual(stopTriggered, true);
  });

  // 21. ChatWorkspace: renders citation rejection banner when answer verification fails
  it('21. renders citation rejection banner when answer verification fails', () => {
    resetDom();
    const rejectedMessages = [
      {
        id: 'msg_rej',
        role: 'assistant',
        content: 'The generated answer could not be verified because it referenced invalid source citations.',
        isStreaming: false,
        isVerified: false,
        isRejected: true,
        rejectionReason: 'Fabricated citation [Source 99]',
        citations: [],
      },
    ];

    render(
      <ChatWorkspace
        messages={rejectedMessages}
        loading={false}
        retrievalMode="hybrid"
        onSelectRetrievalMode={() => {}}
        onSubmitQuestion={() => {}}
        documentCount={1}
      />
    );

    assert.ok(screen.getByText(/Citation Verification Rejection:/i));
    assert.ok(screen.getByText(/The provisional answer referenced invalid, fabricated, or excluded source citations/i));
  });

  // 22. ChatWorkspace: renders verified citations bar only after post-stream validation succeeds
  it('22. renders verified citations bar only after post-stream validation succeeds', () => {
    resetDom();
    const verifiedMessages = [
      {
        id: 'msg_ver',
        role: 'assistant',
        content: 'According to physics, entropy increases [Source 1].',
        isStreaming: false,
        isVerified: true,
        citations: [
          {
            source_id: '[Source 1]',
            filename: 'physics.txt',
            page: 1,
            quote: 'entropy increases',
          },
        ],
        metadata: {
          retrieval_mode: 'hybrid',
          latency_ms: 250,
          context_chunks_used: 1,
          total_chunks_retrieved: 5,
        },
      },
    ];

    render(
      <ChatWorkspace
        messages={verifiedMessages}
        loading={false}
        retrievalMode="hybrid"
        onSelectRetrievalMode={() => {}}
        onSubmitQuestion={() => {}}
        documentCount={1}
      />
    );

    assert.ok(screen.getByText(/Verified Source Citations/i));
    assert.ok(screen.getByText('physics.txt'));
    assert.ok(screen.getByText('Provenanced'));
  });

  // 23. Sidebar: renders conversation history list with titles and active selection highlight
  it('23. renders conversation history list with titles and active selection highlight', () => {
    resetDom();
    const mockConversations = [
      { id: 'c1', title: 'Transformers in NLP', message_count: 4, updated_at: '2026-11-23T11:00:00Z' },
      { id: 'c2', title: 'Thermodynamics Laws', message_count: 2, updated_at: '2026-11-23T11:30:00Z' },
    ];

    render(
      <Sidebar
        documents={[]}
        conversations={mockConversations}
        currentConversationId="c1"
        onSelectConversation={() => {}}
        onNewChat={() => {}}
        onDeleteConversation={() => {}}
      />
    );

    assert.ok(screen.getByText('Transformers in NLP'));
    assert.ok(screen.getByText('Thermodynamics Laws'));
    assert.ok(screen.getByText('4'));
    assert.ok(screen.getByText('2'));
    assert.ok(screen.getByText('2 chats'));
  });

  // 24. Sidebar: clicking New Chat button invokes onNewChat
  it('24. clicking New Chat button invokes onNewChat callback', () => {
    resetDom();
    let newChatCalled = false;

    render(
      <Sidebar
        documents={[]}
        conversations={[]}
        onSelectConversation={() => {}}
        onNewChat={() => { newChatCalled = true; }}
        onDeleteConversation={() => {}}
      />
    );

    const newChatBtn = screen.getByRole('button', { name: /New Chat/i });
    fireEvent.click(newChatBtn);

    assert.strictEqual(newChatCalled, true);
  });

  // 25. Sidebar: selecting a conversation invokes onSelectConversation with conversation ID
  it('25. selecting a conversation invokes onSelectConversation with conversation ID', () => {
    resetDom();
    let selectedId = null;
    const mockConversations = [
      { id: 'c_abc', title: 'Vector Retrieval Architecture', message_count: 3 },
    ];

    render(
      <Sidebar
        documents={[]}
        conversations={mockConversations}
        onSelectConversation={(id) => { selectedId = id; }}
        onNewChat={() => {}}
        onDeleteConversation={() => {}}
      />
    );

    const convItem = screen.getByText('Vector Retrieval Architecture');
    fireEvent.click(convItem);

    assert.strictEqual(selectedId, 'c_abc');
  });

  // 26. Sidebar: conversation delete prompt requires confirmation and triggers onDeleteConversation
  it('26. conversation delete prompt requires confirmation and triggers onDeleteConversation', async () => {
    resetDom();
    let deletedId = null;
    const mockConversations = [
      { id: 'c_del_test', title: 'Session To Delete', message_count: 1 },
    ];

    render(
      <Sidebar
        documents={[]}
        conversations={mockConversations}
        onSelectConversation={() => {}}
        onNewChat={() => {}}
        onDeleteConversation={async (id) => { deletedId = id; }}
      />
    );

    // Initial state: delete trash button is present
    const deleteBtn = screen.getByTitle('Delete conversation');
    assert.ok(deleteBtn);

    // Click delete -> shows confirmation "Delete chat?" and "Yes" / "No"
    fireEvent.click(deleteBtn);
    assert.ok(screen.getByText('Delete chat?'));

    const confirmYesBtn = screen.getByRole('button', { name: 'Yes' });
    await act(async () => {
      fireEvent.click(confirmYesBtn);
    });

    assert.strictEqual(deletedId, 'c_del_test');
  });

});
