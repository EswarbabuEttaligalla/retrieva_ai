# Retrieva AI — Intelligent Document Retrieval & Question Answering System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646cff.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4+-38bdf8.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Retrieva AI is an engineered, production-ready **Retrieval-Augmented Generation (RAG)** application built from the ground up without pre-packaged RAG wrappers (no LangChain, no LlamaIndex, no Haystack). Every pipeline stage—PDF parsing, semantic chunking, embedding generation, vector similarity indexing, context ranking, and grounded generation—is crafted transparently in modular Python and connected to a modern React frontend.

---

## 🎯 Objectives

1. **Understand RAG Internals**: Deconstruct RAG mechanics down to vector math, token budgets, and similarity ranking.
2. **Clean Architecture & Separation of Concerns**: Strictly isolate API controllers, configuration, document ingestion, vector storage, retrieval logic, and presentation.
3. **Robust Engineering & Testing**: High test coverage with Pytest, automated integration probes, and deterministic schemas.
4. **Professional Developer Experience**: Environment-driven settings via Pydantic, structured logging, and an intuitive UI.

---

## 🛠 Technology Stack

### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI (Asynchronous ASGI Web Framework)
- **Settings & Validation**: Pydantic v2 & Pydantic-Settings
- **Server**: Uvicorn with standard workers
- **Testing**: Pytest, Pytest-Asyncio, HTTPX

### Core RAG Engine (Phased Implementation)
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Text Chunking**: Custom sliding-window token/character chunker with overlap
- **Dense Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Storage**: ChromaDB (Embedded local persistent vector store)
- **Similarity Computation**: Custom cosine similarity and vector ranking
- **LLM Generation**: OpenAI API (`gpt-4o-mini` / `gpt-3.5-turbo`)

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS
- **Icons**: Lucide React

---

## 🏛 System Architecture

Retrieva AI is divided into clear functional layers:

```
retrieva-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   ├── health.py          # Service health and diagnostics
│   │   │       │   ├── documents.py       # Document upload, catalog, and deletion
│   │   │       │   ├── retrieval.py       # Standalone retrieval and search diagnostics
│   │   │       │   ├── answer.py          # Grounded QA endpoints (synchronous & SSE streaming)
│   │   │       │   └── conversations.py   # Multi-turn persistent sessions and history
│   │   │       └── router.py              # Aggregated v1 route handler
│   │   ├── conversations/
│   │   │   ├── database.py                # Thread-safe SQLite connection factory with WAL & FK
│   │   │   ├── models.py                  # Pydantic schemas for conversations and messages
│   │   │   ├── repository.py              # SQL CRUD repository with cascading message deletion
│   │   │   ├── query_rewriter.py          # Multi-turn contextual query rewriter with safe fallback
│   │   │   └── service.py                 # Conversation lifecycle & auto-title coordinator
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic BaseSettings singleton & limits
│   │   │   ├── logging.py                 # Structured system logger
│   │   │   └── sse.py                     # W3C-compliant SSE frame formatter
│   │   ├── document_processing/
│   │   │   ├── extractors.py              # PyMuPDF and UTF-8/fallback TXT extractors
│   │   │   ├── cleaner.py                 # NFKC normalization and whitespace cleanup
│   │   │   ├── chunker.py                 # Sliding-window token/character chunker
│   │   │   └── identifier.py              # SHA-256 fingerprinting & catalog registry
│   │   ├── embeddings/
│   │   │   ├── embedder.py                # SentenceTransformers (all-MiniLM-L6-v2)
│   │   │   └── validator.py               # 384-d L2 normalization guardrails
│   │   ├── vector_store/
│   │   │   ├── client.py                  # Persistent ChromaDB client singleton
│   │   │   └── store.py                   # Collection indexing, query & metadata
│   │   ├── retrieval/
│   │   │   ├── bm25.py                    # Exact token BM25 keyword indexer
│   │   │   ├── fusion.py                  # Reciprocal Rank Fusion (RRF) algorithm
│   │   │   ├── reranker.py                # Cross-Encoder neural reranker
│   │   │   └── retriever.py               # Orchestrator (Dense, BM25, Hybrid)
│   │   ├── llm/
│   │   │   ├── client.py                  # OpenAI API client with streaming & retries
│   │   │   ├── context_builder.py         # Dynamic token budgeter & context assembler
│   │   │   ├── prompt_builder.py          # Untrusted data framing & streaming prompt templates
│   │   │   ├── citation_validator.py      # Provenance validation & safe rejection
│   │   │   └── token_counter.py           # Tiktoken exact counting & fallback logic
│   │   ├── schemas/
│   │   │   ├── health.py                  # Health check request/response schemas
│   │   │   ├── document.py                # Upload, catalog, chunk, and delete schemas
│   │   │   ├── retrieval.py               # Search request, candidate & chunk schemas
│   │   │   └── answer.py                  # RAG answer, citation & diagnostic schemas
│   │   ├── services/
│   │   │   ├── ingestion_service.py       # Document processing & vector ingestion
│   │   │   └── rag_service.py             # 10-step RAG pipeline orchestrator (sync & SSE streaming)
│   │   └── main.py                        # FastAPI application factory & lifespan
│   ├── tests/                             # 11 test modules (235 tests collected)
│   ├── requirements.txt                   # Backend dependencies
│   └── .env.example                       # Backend environment template
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.js                  # Fetch API wrapper for all backend routes
│   │   ├── components/
│   │   │   ├── Sidebar.jsx                # Dropzone, upload progress & doc manager
│   │   │   ├── ChatWorkspace.jsx          # Chat dock, starter chips & message cards
│   │   │   ├── MarkdownRenderer.jsx       # XSS-safe markdown with copyable code blocks
│   │   │   ├── CitationDrawer.jsx         # Slide-over source provenance inspector
│   │   │   └── RetrievalInspector.jsx     # Interactive retrieval diagnostics console
│   │   ├── App.jsx                        # Unified responsive application orchestrator
│   │   ├── main.jsx                       # React DOM mount point
│   │   └── index.css                      # Tailwind base and design tokens
│   ├── tests/
│   │   ├── client.test.mjs                # Frontend API client test suite (16 tests)
│   │   └── components.test.jsx            # Frontend component verification suite (26 tests)
│   ├── package.json                       # Frontend dependencies & scripts
│   ├── tailwind.config.js                 # Tailwind CSS configuration
│   ├── vite.config.js                     # Vite build configuration
│   └── index.html                         # SPA HTML entry point
├── docs/
│   └── LEARNING_GUIDE.md                  # Comprehensive educational guide to RAG
├── .env.example                           # Root environment variable template
├── .gitignore                             # Git ignore rules
├── pytest.ini                             # Pytest test discovery & execution settings
├── requirements.txt                       # Root backend dependencies reference
└── README.md                              # Project documentation
```

---

## 📋 Prerequisites

Before running the project, verify that your machine has the following tools installed:

1. **Python**: Version 3.10 or higher (`python --version`)
2. **Node.js**: Version 18.0 or higher (`node --version`)
3. **NPM**: Version 9.0 or higher (`npm --version`)
4. **Git**: (`git --version`)

---

## 🚀 Installation & Setup

### 1. Clone or Open the Repository
```bash
cd rag
```

### 2. Backend Setup

Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:
```bash
pip install -r requirements.txt
```

Create your local environment file:
```bash
cp .env.example .env
```

### 3. Frontend Setup

Navigate to the frontend folder and install dependencies:
```bash
cd frontend
npm install
cd ..
```

---

## 🏃 Running the Application

### Start the Backend Server

From the project root (with `.venv` active):
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger API Docs: `http://127.0.0.1:8000/api/v1/docs`
- Redoc API Docs: `http://127.0.0.1:8000/api/v1/redoc`
- Health Probe: `http://127.0.0.1:8000/api/v1/health`

### Start the Frontend Application

In a separate terminal window:
```bash
cd frontend
npm run dev
```
- Frontend UI: `http://localhost:5173`

---

## 🧪 Testing

Execute the automated test suite with Pytest:

```bash
# Run all tests
pytest -v

# Run tests with output logs
pytest -v -s
```

All tests run against standard FastAPI TestClient and async HTTPX clients.

---

## 📊 Current Development Status

- [x] **Phase 1: Project Initialization & Foundation**
  - Workspace verified clean
  - Modular Clean Architecture established
  - Python virtual environment and dependencies configured
  - FastAPI application factory, lifespan management, structured logging, and CORS setup
  - Health check endpoint (`GET /api/v1/health`) implemented and verified
  - Automated Pytest test suite executed with 100% pass rate
  - React 18 + Vite + Tailwind CSS frontend scaffolded and connected to backend probe
  - Comprehensive architectural and educational documentation created

- [x] **Phase 2: Document Ingestion & Chunking Engine**
  - PyMuPDF integration for PDF text and metadata extraction (multi-page, encrypted, corrupted, empty detection)
  - Resilient TXT extraction with multi-encoding fallback (UTF-8, Latin-1, CP1252)
  - Text normalization pipeline (Unicode NFKC, control char removal, hyphenation repair, whitespace collapse)
  - Custom character-based sliding-window chunker with semantic hierarchy (paragraphs, sentences, words)
  - Exact offset tracking (`start_offset`, `end_offset`) and lossless page reconstruction
  - SHA-256 cryptographic duplicate detection and in-memory document registry
  - Comprehensive 19-test automated suite executed with 100% pass rate
  - Educational guide updated with deep first-principles explanations and walkthroughs

- [x] **Phase 3: Dense Embeddings & Vector Storage**
  - Sentence Transformers integration (`all-MiniLM-L6-v2`) producing 384-dimensional dense vectors
  - L2 vector normalization and finite numerical validation guardrails
  - Persistent ChromaDB vector storage with metadata and chunk provenance preservation
  - Automatic duplicate prevention, document deletion, and collection restart integrity

- [x] **Phase 4: Retrieval Engine & Hybrid Semantic Search**
  - Dense semantic retrieval with cosine distance similarity and score thresholding
  - In-memory BM25 keyword retrieval with IDF weighting and tokenization
  - Reciprocal Rank Fusion (RRF) combining dense and sparse candidate rankings
  - Cross-Encoder neural reranking (`ms-marco-MiniLM-L-6-v2`) for precision relevance
  - 60-chunk, 20-query comparative evaluation benchmark measuring MRR, MAP, and NDCG@K

- [x] **Phase 5: LLM Integration, Prompt Construction & RAG Orchestration**
  - OpenAI client integration (`gpt-4o-mini`) with streaming and error translation
  - Dynamic token budgeting and model context window clamping with `tiktoken`
  - Explicit prompt isolation delimiting untrusted retrieved document context
  - Strict provenance-backed citation validation and uncompromising safe rejection of hallucinated sources
  - 10-step manual RAG orchestration pipeline without LangChain or LlamaIndex

- [x] **Phase 6: Frontend Chat Interface & End-to-End System Integration**
  - Full-stack Document Management API: `POST /upload`, `GET /documents`, `DELETE /{document_id}`
  - Automatic multi-engine synchronization: BM25 index instantly syncs on upload and deletion
  - Drag-and-drop document upload interface with progress bar, file size guards, and catalog stats
  - Real-time retrieval mode selection dock: Dense Semantic, BM25 Keyword, and Hybrid Fusion
  - Interactive chat workspace with starter questions, latency badges, and citation chips
  - Slide-over Citation Drawer displaying chunk metadata, score, rank, and verbatim source excerpts
  - XSS-safe markdown renderer supporting syntax-highlighted code blocks with copy-to-clipboard
  - Comprehensive automated test suite: 150 backend tests (149 passed, 1 skipped) and 17 frontend tests (166 passed, 0 failed, 1 skipped across 167 collected tests)

- [x] **Phase 7, Milestone 1: Standalone Retrieval API & Interactive Retrieval Inspector**
  - Dedicated standalone retrieval endpoint: `POST /api/v1/retrieval/search` (with alias `POST /api/v1/retrieval`)
  - Guaranteed zero OpenAI API invocation or LLM token expenditure during retrieval diagnostics
  - Support for multi-stage scoring metrics: Cosine Distance/Similarity, BM25 Score, RRF Score, and Cross-Encoder Logits
  - Interactive frontend Retrieval Inspector panel with real-time parameter tuning (Strategy mode, Top-K, Candidate-K, Cross-Encoder toggle, Document filters)
  - Latency tracking, candidate chunk cards with verbatim excerpts, and multi-stage score badges
  - 12 automated backend integration tests in `test_retrieval_api.py` covering all Milestone 1 verification scenarios
  - 7 newly added frontend tests (2 API client tests + 5 component tests) in `client.test.mjs` and `components.test.jsx`
  - Total automated test suite: 162 backend tests (161 passed, 1 skipped) + 24 frontend tests (24 passed) = 186 collected, 185 passed, 0 failed, 1 skipped

- [x] **Phase 7, Milestone 2: Real-Time Answer Streaming with SSE, Async Socket I/O & Resource Lifecycle Security**
  - High-performance Server-Sent Events endpoint: `POST /api/v1/answer/stream` with W3C SSE framing and Starlette `StreamingResponse`
  - Non-blocking asynchronous token streaming: `OpenAIClient.generate_stream_async` backed by `openai.AsyncOpenAI` with synchronized thread-pool fallback (`asyncio.to_thread(_safe_next)`), preventing event-loop freezes during upstream network reads
  - Resilient resource lifecycle: thread-safe synchronization (`threading.Lock`) guarding against concurrent `next()` and `close()` collisions (`ValueError: generator already executing`) during client disconnection, paired with idempotent `OpenAIClient.aclose()` in FastAPI lifespan shutdown
  - Intermediate pipeline stage notifications (`retrieving`, `reranking`, `building_context`, `generating`, `verifying`)
  - Strict post-stream citation provenance validation: verifies all inline `[Source N]` references against admitted context chunks before committing verification (guaranteeing source identification validity without asserting semantic factual entailment)
  - Uncompromising anti-hallucination safe rejection: replaces provisional generated text with safe fallback prose if citations reference fabricated or excluded chunks
  - Deterministic event lifecycle: `status` -> `token`* -> `status` -> `citations` | `rejection` | `error` -> `done`
  - Client-side streaming support with native `fetch`, `ReadableStream`, `TextDecoder`, and `AbortController` cancellation
  - Interactive UI controls: streaming status indicator, unverified provisional badge, stop generation button, and rejection banners
  - 30 automated backend tests in `test_streaming.py` verifying protocol compliance, token order, error translation, non-blocking concurrency, worker thread cancellation, lock safety, and lifespan cleanup
  - 9 frontend tests (5 client streaming tests + 4 component tests) verifying SSE parsing, chunk fragmentation, rejection routing, and stop button

- [x] **Phase 7, Milestone 3: Persistent SQLite Conversations, Multi-Turn Chat Memory & Conversational Query Rewriting**
  - Persistent SQLite storage for multi-turn conversations and message history located at `chroma_data/conversation_history.db`
  - Robust relational schema with foreign-key enforcement (`PRAGMA foreign_keys = ON;`), Write-Ahead Logging (`PRAGMA journal_mode = WAL;`), and connection busy timeouts
  - Dedicated access layer in `backend/app/conversations/` (`database.py`, `models.py`, `repository.py`, `query_rewriter.py`, `service.py`, `locks.py`)
  - Non-blocking asynchronous SQLite execution: all synchronous database queries are offloaded via `asyncio.to_thread` across FastAPI endpoints, streaming generators, and RAG service methods, keeping the asyncio event loop responsive under database latency
  - Per-conversation request serialization: `ConversationLockManager` provides granular session-level locks ensuring strict FIFO turn ordering and response pairing for concurrent requests targeting the same conversation, while allowing different conversations to process concurrently without contention
  - Contextual query rewriting using OpenAI (`rewrite_query`, `rewrite_query_async`) resolving pronouns, elisions, and conversational dependencies into standalone search queries
  - Prompt-injection defense & sanitization: untrusted conversation history is isolated within boundary tags (`<conversation_history>`, `<latest_user_question>`), system instructions enforce negative constraints against adversarial directives, and output post-processing enforces strict length limits, quote stripping, and fallback to the raw query upon encountering suspicious injection phrases
  - Resilient query rewriter safeguards: automatic timeout fallback, provider error fallback, empty output fallback, and windowed conversation history
  - Strict document provenance and non-contamination guarantee: mitigating iterative hallucination feedback loops by strictly excluding past conversation turns from the authoritative document evidence pool; conversation history is exclusively used for query reformulation and dialogue continuity
  - End-to-end multi-turn RAG integration: `RAGService.answer_question` and `answer_question_stream` support conversational turns, logging user queries, raw and rewritten queries, and verified assistant answers
  - Safe generation persistence: provisional streaming tokens or cancelled generation runs are never committed as verified assistant responses
  - RESTful conversation API: `POST /conversations`, `GET /conversations`, `GET /conversations/{id}`, `DELETE /conversations/{id}`, `POST /conversations/{id}/answer`, and `POST /conversations/{id}/answer/stream`
  - Responsive frontend session management: Sidebar conversation list with active highlights, "+ New Chat" button, message badges, two-step deletion confirmation, and history restoration
  - 43 automated backend tests in `test_conversations.py` covering schema, repository, rewriter, non-contamination, cancellation, API routes, event loop responsiveness, per-conversation lock serialization, and prompt injection defense
  - 9 newly added frontend tests (5 client API tests + 4 component tests) in `client.test.mjs` and `components.test.jsx` (42 total frontend tests)
  - Total automated test suite: 235 backend tests (234 passed, 0 failed, 1 skipped) + 42 frontend tests (42 passed, 0 failed) = 277 collected, 276 passed, 0 failed, 1 skipped (100% of executed tests passed)
