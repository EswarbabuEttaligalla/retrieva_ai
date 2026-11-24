import React, { useState } from 'react';
import { 
  X, 
  FileText, 
  Copy, 
  Check, 
  ExternalLink, 
  Hash, 
  Bookmark, 
  Sparkles 
} from 'lucide-react';

/**
 * Slide-over drawer / modal displaying verified citation details,
 * source provenance, page references, and verbatim chunk text.
 */
export default function CitationDrawer({ isOpen, citation, onClose }) {
  const [copied, setCopied] = useState(false);

  if (!isOpen || !citation) return null;

  const handleCopyQuote = () => {
    const textToCopy = citation.quote || citation.chunk_text || citation.text || '';
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const sourceId = citation.source_id || 'Source Citation';
  const filename = citation.filename || 'Unknown Document';
  const page = citation.page || citation.page_number || (citation.page_references && citation.page_references.length > 0 ? citation.page_references.join(', ') : 1);
  const quote = citation.quote || citation.chunk_text || citation.text || 'No verbatim quote available.';
  const chunkId = citation.chunk_id || '';
  const score = citation.score !== undefined ? citation.score : null;
  const rank = citation.rank !== undefined ? citation.rank : null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-slate-950/70 backdrop-blur-sm transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Slide-over panel */}
      <div className="relative w-full max-w-lg bg-slate-900 border-l border-slate-800 shadow-2xl flex flex-col h-full z-10 animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="px-6 py-5 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/90 sticky top-0">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Bookmark className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <span>{sourceId}</span>
                <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Verified Source
                </span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5 truncate max-w-xs">{filename}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition"
            title="Close drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {/* Provenance Metadata Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
              <div className="text-[11px] font-mono text-slate-500 flex items-center space-x-1">
                <FileText className="w-3.5 h-3.5" />
                <span>DOCUMENT</span>
              </div>
              <p className="text-xs font-semibold text-slate-200 mt-1 truncate" title={filename}>
                {filename}
              </p>
            </div>

            <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
              <div className="text-[11px] font-mono text-slate-500 flex items-center space-x-1">
                <Hash className="w-3.5 h-3.5" />
                <span>PAGE PROVENANCE</span>
              </div>
              <p className="text-xs font-semibold text-slate-200 mt-1">
                Page {page}
              </p>
            </div>

            {chunkId && (
              <div className="col-span-2 bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                <div className="text-[11px] font-mono text-slate-500">CHUNK IDENTIFIER</div>
                <p className="text-xs font-mono text-indigo-300 mt-1 truncate">
                  {chunkId}
                </p>
              </div>
            )}

            {rank !== null && (
              <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                <div className="text-[11px] font-mono text-slate-500">RETRIEVAL RANK</div>
                <p className="text-xs font-semibold text-slate-200 mt-1">
                  #{rank}
                </p>
              </div>
            )}

            {score !== null && (
              <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                <div className="text-[11px] font-mono text-slate-500">RELEVANCE SCORE</div>
                <p className="text-xs font-mono font-semibold text-brand-400 mt-1">
                  {typeof score === 'number' ? score.toFixed(4) : score}
                </p>
              </div>
            )}
          </div>

          {/* Verbatim Excerpt Card */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-brand-400" />
                <span>Verbatim Grounding Context</span>
              </span>
              <button
                onClick={handleCopyQuote}
                className="inline-flex items-center space-x-1 text-xs font-medium text-slate-400 hover:text-slate-200 transition"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Text</span>
                  </>
                )}
              </button>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/90 text-slate-200 text-xs font-serif sm:text-sm leading-relaxed shadow-inner max-h-96 overflow-y-auto whitespace-pre-wrap selection:bg-brand-500/30">
              "{quote}"
            </div>
          </div>

          {/* Verification Notice */}
          <div className="p-3.5 rounded-xl bg-indigo-950/30 border border-indigo-800/40 text-xs text-indigo-300/90 leading-relaxed">
            <p>
              This chunk was retrieved from local ChromaDB vector storage and verified by the citation provenance validator prior to response synthesis.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/90 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
