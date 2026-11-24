import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Sparkles, 
  User, 
  Bot, 
  Layers, 
  Cpu, 
  Clock, 
  FileText, 
  AlertTriangle, 
  HelpCircle, 
  Trash2, 
  Bookmark, 
  ArrowRight,
  RefreshCw,
  Search,
  CheckCircle2,
  SlidersHorizontal,
  Square,
  ShieldCheck,
  ShieldAlert,
  Loader2
} from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';

const STARTER_QUESTIONS = [
  "What are the core architectural mechanisms described in the uploaded documents?",
  "Can you summarize the primary findings and experimental metrics?",
  "What constraints, limitations, or edge cases are highlighted in the text?",
  "Explain how the system handles errors and data validation."
];

export default function ChatWorkspace({
  messages = [],
  loading = false,
  error = null,
  retrievalMode = 'hybrid',
  streamingStatus = null,
  onSelectRetrievalMode,
  onSubmitQuestion,
  onStopGeneration,
  onClearChat,
  onCitationClick,
  onOpenRetrievalInspector,
  documentCount = 0,
}) {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Handle textarea keypress (Enter to submit, Shift+Enter for newline)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    const query = inputText.trim();
    if (!query || loading) return;
    onSubmitQuestion(query);
    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleStarterClick = (question) => {
    setInputText(question);
    textareaRef.current?.focus();
  };

  const handleTextareaChange = (e) => {
    setInputText(e.target.value);
    // Auto-adjust height up to 160px
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  return (
    <main className="flex-1 flex flex-col h-full bg-slate-950 relative overflow-hidden">
      {/* Top Workspace Header */}
      <header className="h-16 border-b border-slate-800 bg-slate-900/40 backdrop-blur-md px-6 flex items-center justify-between z-10 shrink-0">
        <div className="flex items-center space-x-4">
          <div>
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>Grounded QA Console</span>
              {loading && (
                <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-brand-950 border border-brand-800 text-brand-300 animate-pulse">
                  <span className="w-1.5 h-1.5 rounded-full bg-brand-400"></span>
                  <span>Streaming</span>
                </span>
              )}
            </h2>
            <p className="text-[11px] text-slate-400">
              {documentCount > 0 
                ? `Querying against ${documentCount} indexed ${documentCount === 1 ? 'document' : 'documents'}` 
                : 'Upload documents in the sidebar to begin'}
            </p>
          </div>
        </div>

        {/* Retrieval Mode Selector & Action Buttons */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1 p-1 bg-slate-900 border border-slate-800 rounded-xl text-xs">
            <span className="text-[10px] uppercase font-mono text-slate-500 px-2 flex items-center space-x-1">
              <SlidersHorizontal className="w-3 h-3" />
              <span className="hidden sm:inline">MODE:</span>
            </span>

            <button
              onClick={() => onSelectRetrievalMode('dense')}
              className={`px-2.5 py-1 rounded-lg font-medium transition text-xs ${
                retrievalMode === 'dense'
                  ? 'bg-brand-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              title="Dense Semantic Vector Search (Sentence Transformers)"
            >
              DENSE
            </button>

            <button
              onClick={() => onSelectRetrievalMode('bm25')}
              className={`px-2.5 py-1 rounded-lg font-medium transition text-xs ${
                retrievalMode === 'bm25'
                  ? 'bg-brand-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              title="BM25 Keyword Frequency Search"
            >
              BM25
            </button>

            <button
              onClick={() => onSelectRetrievalMode('hybrid')}
              className={`px-2.5 py-1 rounded-lg font-medium transition text-xs ${
                retrievalMode === 'hybrid'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              title="Hybrid Fusion (Dense + BM25 with Reciprocal Rank Fusion & Cross-Encoder Reranker)"
            >
              HYBRID
            </button>
          </div>

          {onOpenRetrievalInspector && (
            <button
              onClick={onOpenRetrievalInspector}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-xs font-medium text-slate-300 hover:text-white transition shadow-xs"
              title="Open standalone Retrieval Inspector to diagnose chunks and score mechanics"
            >
              <Search className="w-3.5 h-3.5 text-indigo-400" />
              <span className="hidden sm:inline">Inspect Retrieval</span>
            </button>
          )}

          {messages.length > 0 && (
            <button
              onClick={onClearChat}
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition"
              title="Clear conversation"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </header>

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {messages.length === 0 && !loading ? (
          /* Empty / Welcome State */
          <div className="max-w-2xl mx-auto py-12 text-center space-y-8 animate-in fade-in duration-300">
            <div className="space-y-3">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-brand-600 to-indigo-600 flex items-center justify-center mx-auto shadow-xl shadow-brand-500/20">
                <Sparkles className="w-7 h-7 text-white" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Retrieva AI Assistant
              </h1>
              <p className="text-sm text-slate-400 leading-relaxed max-w-lg mx-auto">
                Ask questions grounded strictly in your uploaded PDF and TXT documents. Every answer is synthesized with verifiable page citations and anti-hallucination guards.
              </p>
            </div>

            {/* Architecture Highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left">
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
                <div className="text-xs font-semibold text-brand-400 mb-1 flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Strict Grounding</span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Refuses to guess or hallucinate when context is insufficient.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
                <div className="text-xs font-semibold text-indigo-400 mb-1 flex items-center space-x-1.5">
                  <Bookmark className="w-3.5 h-3.5" />
                  <span>Verifiable Sources</span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Inspect exact document filename, page, and chunk provenance.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
                <div className="text-xs font-semibold text-emerald-400 mb-1 flex items-center space-x-1.5">
                  <Layers className="w-3.5 h-3.5" />
                  <span>Hybrid Retrieval</span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Fuses dense embeddings & BM25 with Reciprocal Rank Fusion.
                </p>
              </div>
            </div>

            {/* Starter Questions */}
            {documentCount > 0 && (
              <div className="space-y-3 pt-4 text-left">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                  Suggested Questions
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {STARTER_QUESTIONS.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleStarterClick(q)}
                      className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/80 hover:border-slate-700 hover:bg-slate-900 text-left transition flex items-center justify-between group"
                    >
                      <span className="text-xs text-slate-300 group-hover:text-white line-clamp-2">
                        {q}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-brand-400 shrink-0 ml-2 transition" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Message Stream */
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, index) => {
              const isUser = msg.role === 'user';
              const isStreamingThis = msg.isStreaming;

              return (
                <div
                  key={msg.id || index}
                  className={`flex items-start space-x-3.5 ${
                    isUser ? 'flex-row-reverse space-x-reverse' : ''
                  }`}
                >
                  {/* Avatar */}
                  <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${
                    isUser 
                      ? 'bg-slate-800 text-slate-200 border border-slate-700' 
                      : 'bg-gradient-to-tr from-brand-600 to-indigo-600 text-white shadow-md shadow-brand-500/20'
                  }`}>
                    {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>

                  {/* Message Bubble / Card */}
                  <div className={`max-w-[85%] rounded-2xl ${
                    isUser 
                      ? 'bg-brand-600 text-white px-4 py-3 shadow-md' 
                      : 'bg-slate-900/80 border border-slate-800/90 p-5 shadow-xl backdrop-blur-sm'
                  }`}>
                    {isUser ? (
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div className="space-y-4">
                        {/* Streaming Stage Status Notification */}
                        {isStreamingThis && (
                          <div className="p-2.5 rounded-xl bg-indigo-950/40 border border-indigo-800/50 flex items-center justify-between text-indigo-300 text-xs">
                            <div className="flex items-center space-x-2">
                              <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                              <span>{msg.statusMessage || 'Streaming grounded response...'}</span>
                            </div>
                            <span className="text-[10px] font-mono uppercase text-indigo-400 bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800/40">
                              Provisional
                            </span>
                          </div>
                        )}

                        {/* Citation Rejection Banner */}
                        {msg.isRejected && (
                          <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-start space-x-2.5 text-rose-300">
                            <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                            <div className="text-xs leading-relaxed">
                              <span className="font-semibold">Citation Verification Rejection: </span>
                              The provisional answer referenced invalid, fabricated, or excluded source citations. The unverified draft was discarded and replaced with a verified safe response.
                            </div>
                          </div>
                        )}

                        {/* Insufficient Context Warning Banner */}
                        {msg.insufficient_context && (
                          <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/60 flex items-start space-x-2.5 text-amber-300">
                            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                            <div className="text-xs leading-relaxed">
                              <span className="font-semibold">Insufficient Document Context: </span>
                              The ingested knowledge base did not contain enough substantiated information to completely answer this query.
                            </div>
                          </div>
                        )}

                        {/* User Stopped Banner */}
                        {msg.isStopped && (
                          <div className="p-2.5 rounded-xl bg-slate-800/50 border border-slate-700/60 flex items-center space-x-2 text-slate-400 text-xs">
                            <Square className="w-3 h-3 text-slate-400 fill-current" />
                            <span>Generation stopped by user before citation validation completed (unverified).</span>
                          </div>
                        )}

                        {/* Answer Text */}
                        <div className="prose prose-invert max-w-none text-sm">
                          {msg.content ? (
                            <MarkdownRenderer
                              content={msg.content}
                              onCitationClick={(sourceId) => {
                                const matched = msg.citations?.find(c => c.source_id === sourceId);
                                if (matched && onCitationClick) {
                                  onCitationClick(matched);
                                }
                              }}
                            />
                          ) : (
                            <span className="text-slate-500 italic text-xs">Waiting for tokens...</span>
                          )}
                          {isStreamingThis && (
                            <span className="inline-block w-1.5 h-3.5 bg-brand-400 ml-1 animate-pulse align-middle" />
                          )}
                        </div>

                        {/* Citations Bar (Only after validation passes) */}
                        {msg.isVerified && msg.citations && msg.citations.length > 0 && (
                          <div className="pt-3 border-t border-slate-800/80 space-y-2">
                            <div className="flex items-center space-x-1.5 text-[11px] font-mono text-emerald-400 uppercase tracking-wider">
                              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                              <span>Verified Source Citations ({msg.citations.length})</span>
                            </div>

                            <div className="flex flex-wrap gap-2">
                              {msg.citations.map((cit, cIdx) => {
                                const pageDisplay = cit.page || cit.page_number || (cit.page_references && cit.page_references[0]) || 1;
                                return (
                                  <button
                                    key={cIdx}
                                    onClick={() => onCitationClick && onCitationClick(cit)}
                                    className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-950/80 border border-slate-800 hover:border-brand-500/60 hover:bg-slate-800/60 text-slate-300 hover:text-white transition group"
                                    title={`Inspect ${cit.source_id}: ${cit.filename}, page ${pageDisplay}`}
                                  >
                                    <span className="font-mono text-brand-400 font-semibold text-[11px]">
                                      [{cit.source_id}]
                                    </span>
                                    <span className="truncate max-w-[140px] text-slate-300">
                                      {cit.filename}
                                    </span>
                                    <span className="text-[10px] text-slate-400">p.{pageDisplay}</span>
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Diagnostic Metadata Tray */}
                        {msg.metadata && Object.keys(msg.metadata).length > 0 && (
                          <div className="pt-2 border-t border-slate-800/50 flex flex-wrap items-center gap-3 text-[10px] font-mono text-slate-400">
                            <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 uppercase">
                              {msg.metadata.retrieval_mode || retrievalMode}
                            </span>

                            {msg.ttft_ms !== undefined && (
                              <span className="flex items-center space-x-1 text-indigo-400">
                                <span>TTFT: {msg.ttft_ms} ms</span>
                              </span>
                            )}

                            {msg.metadata.latency_ms && (
                              <span className="flex items-center space-x-1">
                                <Clock className="w-3 h-3 text-slate-400" />
                                <span>{msg.metadata.latency_ms} ms</span>
                              </span>
                            )}

                            {msg.metadata.context_chunks_used !== undefined && (
                              <span>
                                {msg.metadata.context_chunks_used} / {msg.metadata.total_chunks_retrieved || 0} chunks used
                              </span>
                            )}

                            {msg.isVerified && (
                              <span className="text-emerald-400 flex items-center space-x-1">
                                <CheckCircle2 className="w-3 h-3" />
                                <span>Provenanced</span>
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* In-flight Loading Animation (when query is starting and no assistant msg yet) */}
            {loading && messages.filter(m => m.role === 'assistant' && m.isStreaming).length === 0 && (
              <div className="flex items-start space-x-3.5">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 text-white flex items-center justify-center shrink-0 shadow-md">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="bg-slate-900/80 border border-slate-800/90 rounded-2xl p-4 shadow-xl flex items-center space-x-3 text-slate-300 text-xs">
                  <RefreshCw className="w-4 h-4 text-brand-400 animate-spin shrink-0" />
                  <div className="space-y-0.5">
                    <p className="font-medium text-slate-200">
                      Searching knowledge base & synthesizing answer...
                    </p>
                    <p className="text-[11px] text-slate-400">
                      Filtering candidates via {retrievalMode.toUpperCase()} retrieval & verifying citations.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mx-4 sm:mx-8 mb-2 p-3 rounded-xl bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs flex items-center justify-between z-20">
          <div className="flex items-center space-x-2 truncate">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span className="truncate">{error}</span>
          </div>
          <button
            onClick={() => handleSubmit()}
            className="ml-3 px-2 py-0.5 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-200 text-[11px] font-medium shrink-0"
          >
            Retry
          </button>
        </div>
      )}

      {/* Bottom Sticky Input Dock */}
      <footer className="p-4 sm:px-8 border-t border-slate-800/80 bg-slate-900/60 backdrop-blur-md shrink-0">
        <div className="max-w-3xl mx-auto">
          <div className="relative flex items-end rounded-2xl bg-slate-900 border border-slate-800 focus-within:border-brand-500/80 shadow-2xl transition">
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputText}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder={
                documentCount > 0 
                  ? "Ask a question about your documents... (Enter to send, Shift+Enter for newline)" 
                  : "Upload a document in the sidebar to start asking questions..."
              }
              disabled={loading}
              className="w-full py-3.5 pl-4 pr-24 bg-transparent text-sm text-slate-100 placeholder-slate-500 resize-none focus:outline-none max-h-40 overflow-y-auto leading-relaxed"
            />

            <div className="absolute right-2.5 bottom-2.5 flex items-center space-x-1.5">
              {loading && onStopGeneration ? (
                <button
                  type="button"
                  onClick={onStopGeneration}
                  className="px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow-md shadow-rose-600/30 transition"
                  title="Stop generating"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={!inputText.trim() || loading}
                  className={`p-2 rounded-xl transition ${
                    inputText.trim() && !loading
                      ? 'bg-brand-600 hover:bg-brand-500 text-white shadow-md shadow-brand-500/20'
                      : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  }`}
                  title="Send question"
                >
                  <Send className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between mt-2 px-1 text-[11px] text-slate-400">
            <span>Ground-up RAG pipeline • Real-time SSE streaming • Citation-safe</span>
            <span>Enter ↵ to send</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
