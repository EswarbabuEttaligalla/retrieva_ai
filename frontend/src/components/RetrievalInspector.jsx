import React, { useState } from 'react';
import {
  Search,
  Sliders,
  X,
  Clock,
  Layers,
  Database,
  CheckCircle2,
  AlertCircle,
  FileText,
  ChevronRight,
  Filter,
  Zap,
  Sparkles,
  RefreshCw,
  Hash,
  Activity
} from 'lucide-react';
import { searchRetrieval } from '../api/client';

export default function RetrievalInspector({
  isOpen = false,
  onClose,
  documents = [],
}) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [topK, setTopK] = useState(5);
  const [candidateK, setCandidateK] = useState(20);
  const [enableReranking, setEnableReranking] = useState(false);
  const [docFilter, setDocFilter] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  if (!isOpen) return null;

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery) return;

    setLoading(true);
    setError(null);

    try {
      const data = await searchRetrieval({
        query: cleanQuery,
        mode,
        top_k: Number(topK),
        candidate_k: Number(candidateK),
        enable_reranking: enableReranking,
        document_id_filter: docFilter || null,
      });
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to execute retrieval search.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-950/75 backdrop-blur-xs transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-over panel */}
      <div className="relative w-full max-w-3xl bg-slate-900 border-l border-slate-800 shadow-2xl flex flex-col h-full z-10 text-slate-100">
        {/* Header */}
        <div className="p-6 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Sliders className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center space-x-2">
                <span>Retrieval Inspector</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono font-medium">
                  DIAGNOSTIC
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Inspect retrieved chunks and scoring mechanics without LLM token usage
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition"
            aria-label="Close Inspector"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search Controls Form */}
        <form onSubmit={handleSearch} className="p-6 border-b border-slate-800 bg-slate-950/40 space-y-4 shrink-0">
          {/* Query Input */}
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter search query to inspect retrieval candidates..."
              className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-3 pl-10 pr-24 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-inner"
              autoFocus
            />
            <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2 top-2 bottom-2 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:pointer-events-none text-xs font-semibold text-white transition flex items-center space-x-1.5 shadow-sm"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Zap className="w-3.5 h-3.5" />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>

          {/* Configuration Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            {/* Mode Selector */}
            <div>
              <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                Strategy Mode
              </label>
              <div className="grid grid-cols-3 gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
                {['dense', 'bm25', 'hybrid'].map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={`py-1.5 rounded-lg font-medium text-center uppercase text-[11px] transition ${
                      mode === m
                        ? 'bg-indigo-600 text-white font-bold shadow-xs'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Top-K Selector */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  Top-K Return
                </label>
                <span className="font-mono text-indigo-400 font-bold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
              />
            </div>

            {/* Candidate-K Selector */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  Candidate Pool
                </label>
                <span className="font-mono text-indigo-400 font-bold">{candidateK}</span>
              </div>
              <input
                type="range"
                min="5"
                max="50"
                value={candidateK}
                onChange={(e) => setCandidateK(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
              />
            </div>

            {/* Document Filter */}
            <div>
              <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                Document Filter
              </label>
              <select
                value={docFilter}
                onChange={(e) => setDocFilter(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
              >
                <option value="">All Documents</option>
                {documents.map((doc) => (
                  <option key={doc.document_id} value={doc.document_id}>
                    {doc.filename || doc.document_id}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Cross-Encoder Toggle */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center space-x-2">
              <input
                id="inspector-rerank"
                type="checkbox"
                checked={enableReranking}
                onChange={(e) => setEnableReranking(e.target.checked)}
                className="w-4 h-4 rounded text-indigo-600 focus:ring-indigo-500 focus:ring-offset-slate-900 bg-slate-800 border-slate-700"
              />
              <label htmlFor="inspector-rerank" className="text-xs text-slate-300 cursor-pointer select-none">
                Enable Cross-Encoder Reranking (<code className="text-indigo-300 font-mono text-[10px]">ms-marco-MiniLM-L-6-v2</code>)
              </label>
            </div>
          </div>
        </form>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Error Alert */}
          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-start space-x-3 text-sm">
              <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
              <div>
                <p className="font-semibold text-white">Retrieval Search Error</p>
                <p className="text-xs mt-0.5 text-rose-300/90">{error}</p>
              </div>
            </div>
          )}

          {/* Diagnostic Metrics Banner (when results available) */}
          {results && (
            <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-800 flex flex-wrap gap-4 items-center justify-between text-xs">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1.5 text-slate-400">
                  <Clock className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Latency:</span>
                  <span className="font-mono font-bold text-white">{results.execution_time_ms} ms</span>
                </div>
                <div className="flex items-center space-x-1.5 text-slate-400">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Candidates:</span>
                  <span className="font-mono font-bold text-white">{results.total_candidates_found} found</span>
                </div>
                <div className="flex items-center space-x-1.5 text-slate-400">
                  <Database className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Returned:</span>
                  <span className="font-mono font-bold text-emerald-400">{results.returned_chunks_count} chunks</span>
                </div>
              </div>

              {results.reranking_summary && (
                <div className="flex items-center space-x-2 text-[11px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 px-2.5 py-1 rounded-lg">
                  <Sparkles className="w-3 h-3 text-indigo-400" />
                  <span>Reranked {results.reranking_summary.candidates_in} → {results.reranking_summary.candidates_out} ({results.reranking_summary.duration_ms} ms)</span>
                </div>
              )}
            </div>
          )}

          {/* Results List */}
          {results && results.chunks && results.chunks.length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                <span>Ranked Candidate Chunks</span>
                <span>Sorted by Rank Priority</span>
              </h3>

              {results.chunks.map((chunk) => (
                <div
                  key={chunk.chunk_id}
                  className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 transition space-y-3"
                >
                  {/* Chunk Card Header */}
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/60 pb-2.5 text-xs">
                    <div className="flex items-center space-x-2">
                      <span className="w-6 h-6 rounded-lg bg-indigo-600/20 border border-indigo-500/30 font-mono font-bold text-indigo-400 flex items-center justify-center text-xs">
                        #{chunk.rank}
                      </span>
                      <span className="font-medium text-slate-200 flex items-center space-x-1">
                        <FileText className="w-3.5 h-3.5 text-slate-400" />
                        <span className="truncate max-w-[200px]">{chunk.filename || chunk.document_id}</span>
                      </span>
                    </div>

                    <div className="flex items-center space-x-2 text-[11px] text-slate-400">
                      <span className="bg-slate-800 px-2 py-0.5 rounded-md">Page {chunk.page_number}</span>
                      <span className="bg-slate-800 px-2 py-0.5 rounded-md font-mono text-[10px]">
                        Offsets {chunk.start_offset}–{chunk.end_offset}
                      </span>
                    </div>
                  </div>

                  {/* Verbatim Chunk Content */}
                  <div className="text-xs text-slate-300 leading-relaxed bg-slate-900/60 p-3 rounded-lg border border-slate-800/40 font-mono whitespace-pre-wrap select-text">
                    {chunk.chunk_text}
                  </div>

                  {/* Multi-Stage Score Badges */}
                  <div className="flex flex-wrap gap-2 pt-1 text-[11px]">
                    {/* Dense Scores */}
                    {chunk.cosine_similarity !== null && chunk.cosine_similarity !== undefined && (
                      <span
                        className="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center space-x-1"
                        title="Calculated Cosine Similarity (1.0 - cosine_distance)"
                      >
                        <span className="text-[10px] text-slate-400 uppercase font-mono">Cosine Sim:</span>
                        <span className="font-mono font-bold">{chunk.cosine_similarity.toFixed(4)}</span>
                      </span>
                    )}

                    {chunk.cosine_distance !== null && chunk.cosine_distance !== undefined && (
                      <span
                        className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700/60 text-slate-300 flex items-center space-x-1"
                        title="ChromaDB Cosine Distance Metric (0.0 = identical)"
                      >
                        <span className="text-[10px] text-slate-400 uppercase font-mono">Distance:</span>
                        <span className="font-mono font-bold">{chunk.cosine_distance.toFixed(4)}</span>
                      </span>
                    )}

                    {/* BM25 Score */}
                    {chunk.bm25_score !== null && chunk.bm25_score !== undefined && (
                      <span
                        className="px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center space-x-1"
                        title="Okapi BM25 Term Frequency & Length-Normalized IDF Score"
                      >
                        <span className="text-[10px] text-slate-400 uppercase font-mono">BM25:</span>
                        <span className="font-mono font-bold">{chunk.bm25_score.toFixed(4)}</span>
                      </span>
                    )}

                    {/* RRF Score */}
                    {chunk.rrf_score !== null && chunk.rrf_score !== undefined && (
                      <span
                        className="px-2.5 py-1 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center space-x-1"
                        title="Reciprocal Rank Fusion Score: 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)"
                      >
                        <span className="text-[10px] text-slate-400 uppercase font-mono">RRF Score:</span>
                        <span className="font-mono font-bold">{chunk.rrf_score.toFixed(4)}</span>
                      </span>
                    )}

                    {/* Cross-Encoder Reranker Score */}
                    {chunk.cross_encoder_score !== null && chunk.cross_encoder_score !== undefined && (
                      <span
                        className="px-2.5 py-1 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-300 flex items-center space-x-1"
                        title="Cross-Encoder Joint Query-Chunk Attention Logit"
                      >
                        <span className="text-[10px] text-slate-400 uppercase font-mono">Cross-Encoder:</span>
                        <span className="font-mono font-bold">{chunk.cross_encoder_score.toFixed(4)}</span>
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : results ? (
            <div className="p-12 text-center rounded-xl bg-slate-950/40 border border-slate-800 space-y-3">
              <Search className="w-8 h-8 text-slate-600 mx-auto" />
              <h4 className="text-sm font-semibold text-slate-300">No Candidate Chunks Found</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                No indexed chunks matched your query and filter criteria under <span className="font-mono text-indigo-400 uppercase">{mode}</span> mode.
              </p>
            </div>
          ) : (
            <div className="p-12 text-center rounded-xl bg-slate-950/20 border border-dashed border-slate-800 space-y-3">
              <Sliders className="w-8 h-8 text-slate-600 mx-auto" />
              <h4 className="text-sm font-semibold text-slate-400">Ready to Inspect Retrieval</h4>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Submit a query above to directly execute Dense, BM25, or Hybrid retrieval and analyze candidate relevance, ranks, and score metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
