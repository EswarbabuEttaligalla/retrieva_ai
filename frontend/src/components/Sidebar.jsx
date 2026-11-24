import React, { useState, useRef } from 'react';
import { 
  Sparkles, 
  UploadCloud, 
  FileText, 
  File, 
  Trash2, 
  AlertCircle, 
  CheckCircle2, 
  RefreshCw, 
  Server, 
  Layers, 
  ChevronRight,
  Database,
  Info,
  Plus,
  MessageSquare,
  MessagesSquare
} from 'lucide-react';

/**
 * Format bytes into human-readable string (KB, MB).
 */
function formatBytes(bytes, decimals = 1) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format date string into readable local timestamp.
 */
function formatDate(dateString) {
  if (!dateString) return '';
  try {
    const d = new Date(dateString);
    return d.toLocaleDateString(undefined, { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    });
  } catch {
    return dateString;
  }
}

export default function Sidebar({
  documents = [],
  loading = false,
  uploading = false,
  backendLive = false,
  onUpload,
  onDelete,
  onRefresh,
  selectedDocIds = [],
  onToggleDocSelection,
  conversations = [],
  currentConversationId = null,
  onSelectConversation = () => {},
  onNewChat = () => {},
  onDeleteConversation = () => {},
}) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [confirmDeleteConvId, setConfirmDeleteConvId] = useState(null);
  const [deletingConvId, setDeletingConvId] = useState(null);
  const fileInputRef = useRef(null);

  // Drag & drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await processFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      await processFileUpload(e.target.files[0]);
      e.target.value = '';
    }
  };

  const processFileUpload = async (file) => {
    setUploadMessage(null);
    setUploadError(null);

    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (ext !== '.pdf' && ext !== '.txt') {
      setUploadError(`Unsupported file format '${ext}'. Please upload .pdf or .txt files only.`);
      return;
    }

    if (file.size === 0) {
      setUploadError('Uploaded file is empty (0 bytes).');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadError('File exceeds the 10 MB limit.');
      return;
    }

    try {
      const res = await onUpload(file);
      if (res && res.status === 'duplicate') {
        setUploadMessage({
          type: 'duplicate',
          text: `Document '${file.name}' is already indexed in the vector store.`,
        });
      } else {
        setUploadMessage({
          type: 'success',
          text: `Indexed '${file.name}' (${res.total_chunks} chunks).`,
        });
      }
      setTimeout(() => setUploadMessage(null), 5000);
    } catch (err) {
      setUploadError(err.message || 'Failed to upload document.');
    }
  };

  const handleDeleteClick = (docId) => {
    setConfirmDeleteId(docId);
  };

  const handleConfirmDelete = async (docId) => {
    setDeletingId(docId);
    try {
      await onDelete(docId);
      setConfirmDeleteId(null);
    } catch (err) {
      setUploadError(err.message || 'Failed to delete document.');
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteConvClick = (e, convId) => {
    e.stopPropagation();
    setConfirmDeleteConvId(convId);
  };

  const handleConfirmDeleteConv = async (e, convId) => {
    e.stopPropagation();
    setDeletingConvId(convId);
    try {
      await onDeleteConversation(convId);
      setConfirmDeleteConvId(null);
    } catch (err) {
      setUploadError(err.message || 'Failed to delete conversation.');
    } finally {
      setDeletingConvId(null);
    }
  };

  const handleCancelDeleteConv = (e) => {
    e.stopPropagation();
    setConfirmDeleteConvId(null);
  };

  const totalChunksCount = documents.reduce((acc, doc) => acc + (doc.total_chunks || 0), 0);

  return (
    <aside className="w-80 md:w-88 lg:w-96 bg-slate-900 border-r border-slate-800 flex flex-col h-full shrink-0 select-none">
      {/* Top Brand Banner */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-base font-bold text-white tracking-tight">Retrieva AI</span>
              <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-brand-500/10 text-brand-400 border border-brand-500/20">
                v0.1.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Intelligent RAG System</p>
          </div>
        </div>

        {/* Backend live status indicator */}
        <div 
          className={`flex items-center space-x-1.5 px-2 py-1 rounded-md text-[11px] font-medium border ${
            backendLive 
              ? 'bg-emerald-950/40 border-emerald-800/80 text-emerald-400' 
              : 'bg-rose-950/40 border-rose-800/80 text-rose-400'
          }`}
          title={backendLive ? 'FastAPI Backend Operational' : 'FastAPI Backend Offline'}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${backendLive ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
          <span>{backendLive ? 'Online' : 'Offline'}</span>
        </div>
      </div>

      {/* New Chat Action */}
      <div className="p-3 border-b border-slate-800/80 bg-slate-900/40">
        <button
          onClick={onNewChat}
          className="w-full py-2 px-3 rounded-lg bg-brand-600 hover:bg-brand-500 active:scale-[0.99] text-white font-medium text-xs flex items-center justify-center space-x-2 transition shadow-sm"
          title="Start a new chat session"
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Main Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Conversations History Section */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span className="flex items-center space-x-1.5">
              <MessagesSquare className="w-3.5 h-3.5 text-brand-400" />
              <span>CONVERSATIONS</span>
            </span>
            <span className="text-[10px] font-normal text-slate-400">
              {conversations.length} {conversations.length === 1 ? 'chat' : 'chats'}
            </span>
          </div>

          {conversations.length === 0 ? (
            <div className="p-3 rounded-xl border border-dashed border-slate-800 text-center text-[11px] text-slate-500 bg-slate-950/20">
              No conversations yet. Start a new chat!
            </div>
          ) : (
            <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
              {conversations.map((conv) => {
                const isActive = currentConversationId === conv.id;
                const isConfirming = confirmDeleteConvId === conv.id;

                return (
                  <div
                    key={conv.id}
                    onClick={() => onSelectConversation(conv.id)}
                    className={`group relative flex items-center justify-between p-2 rounded-lg cursor-pointer transition text-xs border ${
                      isActive
                        ? 'bg-brand-500/10 border-brand-500/40 text-brand-200 shadow-xs'
                        : 'bg-slate-950/30 border-slate-800/60 text-slate-300 hover:bg-slate-800/60 hover:text-white'
                    }`}
                  >
                    {isConfirming ? (
                      <div className="flex items-center justify-between w-full py-0.5 space-x-2">
                        <span className="text-[11px] text-rose-400 truncate">Delete chat?</span>
                        <div className="flex items-center space-x-1 shrink-0">
                          <button
                            onClick={(e) => handleConfirmDeleteConv(e, conv.id)}
                            disabled={deletingConvId === conv.id}
                            className="px-2 py-0.5 rounded bg-rose-600 hover:bg-rose-500 text-white text-[10px] font-semibold transition"
                          >
                            {deletingConvId === conv.id ? '...' : 'Yes'}
                          </button>
                          <button
                            onClick={handleCancelDeleteConv}
                            className="px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 text-[10px] transition"
                          >
                            No
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-center space-x-2 min-w-0 flex-1">
                          <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-brand-400' : 'text-slate-500 group-hover:text-slate-300'}`} />
                          <span className="truncate font-medium text-[11px]" title={conv.title}>
                            {conv.title || 'Untitled Chat'}
                          </span>
                        </div>
                        <div className="flex items-center space-x-1 shrink-0 ml-1">
                          {conv.message_count !== undefined && conv.message_count > 0 && (
                            <span className="px-1.5 py-0.2 rounded-full text-[9px] bg-slate-800 text-slate-400 font-mono">
                              {conv.message_count}
                            </span>
                          )}
                          <button
                            onClick={(e) => handleDeleteConvClick(e, conv.id)}
                            className="opacity-0 group-hover:opacity-100 p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-slate-800 transition"
                            title="Delete conversation"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
        {/* Upload Zone */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span>INGEST DOCUMENTS</span>
            <span className="text-[10px] font-normal text-slate-400">PDF, TXT (Max 10MB)</span>
          </div>

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
            className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all duration-200 ${
              dragActive 
                ? 'border-brand-500 bg-brand-500/10 scale-[1.01]' 
                : 'border-slate-700/80 hover:border-slate-600 bg-slate-950/40 hover:bg-slate-950/60'
            } ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="w-10 h-10 rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400 group-hover:text-brand-400">
                {uploading ? (
                  <RefreshCw className="w-5 h-5 text-brand-400 animate-spin" />
                ) : (
                  <UploadCloud className="w-5 h-5" />
                )}
              </div>
              <div className="text-xs">
                {uploading ? (
                  <span className="text-brand-400 font-medium">Extracting, chunking & indexing...</span>
                ) : (
                  <>
                    <span className="font-semibold text-slate-200 hover:text-brand-300">
                      Click to upload
                    </span>{' '}
                    <span className="text-slate-400">or drag & drop</span>
                  </>
                )}
              </div>
              <div className="flex items-center space-x-1.5 text-[10px] text-slate-500">
                <span className="px-1.5 py-0.5 rounded bg-slate-800/60 font-mono">.PDF</span>
                <span className="px-1.5 py-0.5 rounded bg-slate-800/60 font-mono">.TXT</span>
              </div>
            </div>
          </div>

          {/* Upload Status / Error Feedback */}
          {uploadMessage && (
            <div className={`p-2.5 rounded-lg text-xs flex items-center space-x-2 border ${
              uploadMessage.type === 'duplicate'
                ? 'bg-amber-950/40 border-amber-800/80 text-amber-300'
                : 'bg-emerald-950/40 border-emerald-800/80 text-emerald-300'
            }`}>
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span className="truncate">{uploadMessage.text}</span>
            </div>
          )}

          {uploadError && (
            <div className="p-2.5 rounded-lg text-xs flex items-center justify-between border bg-rose-950/40 border-rose-800/80 text-rose-300">
              <div className="flex items-center space-x-2 truncate">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span className="truncate">{uploadError}</span>
              </div>
              <button 
                onClick={() => setUploadError(null)} 
                className="text-rose-400 hover:text-rose-200 text-xs ml-2"
              >
                ×
              </button>
            </div>
          )}
        </div>

        {/* Knowledge Base Statistics Badge */}
        <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800 flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2 text-slate-300">
            <Database className="w-4 h-4 text-brand-400" />
            <span className="font-medium">Knowledge Base</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-400 text-[11px]">
            <span>{documents.length} {documents.length === 1 ? 'doc' : 'docs'}</span>
            <span>•</span>
            <span>{totalChunksCount} chunks</span>
          </div>
        </div>

        {/* Ingested Documents List */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <div className="flex items-center space-x-1.5">
              <span>DOCUMENT CATALOG</span>
              <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-slate-800 text-slate-400">
                {documents.length}
              </span>
            </div>
            <button
              onClick={onRefresh}
              disabled={loading}
              className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
              title="Refresh document list"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-brand-400' : ''}`} />
            </button>
          </div>

          {documents.length === 0 ? (
            /* Empty State */
            <div className="p-6 rounded-xl border border-dashed border-slate-800 text-center bg-slate-950/30 space-y-2">
              <FileText className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="text-xs text-slate-400 font-medium">No documents indexed yet</p>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Upload a PDF or TXT document above to start asking grounded questions.
              </p>
            </div>
          ) : (
            /* Document Cards */
            <div className="space-y-2">
              {documents.map((doc) => {
                const isSelected = selectedDocIds.length === 0 || selectedDocIds.includes(doc.document_id);
                const isDeleting = deletingId === doc.document_id;
                const isConfirming = confirmDeleteId === doc.document_id;
                const isPdf = (doc.file_extension || '').toLowerCase() === '.pdf';

                return (
                  <div
                    key={doc.document_id}
                    className={`group relative p-3 rounded-xl border transition-all duration-150 ${
                      isSelected 
                        ? 'bg-slate-950/60 border-slate-800 hover:border-slate-700' 
                        : 'bg-slate-950/20 border-slate-800/40 opacity-60'
                    }`}
                  >
                    {isConfirming ? (
                      /* Confirm deletion prompt */
                      <div className="p-1 text-center space-y-2">
                        <p className="text-xs text-rose-300 font-medium">
                          Delete this document & its vector chunks?
                        </p>
                        <div className="flex items-center justify-center space-x-2">
                          <button
                            onClick={() => handleConfirmDelete(doc.document_id)}
                            disabled={isDeleting}
                            className="px-2.5 py-1 text-xs font-semibold rounded bg-rose-600 hover:bg-rose-500 text-white transition"
                          >
                            {isDeleting ? 'Deleting...' : 'Yes, Delete'}
                          </button>
                          <button
                            onClick={() => setConfirmDeleteId(null)}
                            disabled={isDeleting}
                            className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      /* Standard document card */
                      <div className="flex items-start justify-between space-x-2">
                        <div className="flex items-start space-x-2.5 min-w-0">
                          {/* File Icon */}
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                            isPdf 
                              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' 
                              : 'bg-sky-500/10 text-sky-400 border border-sky-500/20'
                          }`}>
                            <FileText className="w-4 h-4" />
                          </div>

                          {/* Info */}
                          <div className="min-w-0">
                            <h4 
                              className="text-xs font-semibold text-slate-200 truncate hover:text-white transition"
                              title={doc.filename}
                            >
                              {doc.filename}
                            </h4>

                            <div className="flex flex-wrap items-center gap-1.5 mt-1 text-[10px] text-slate-400">
                              <span className="px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 font-mono">
                                {doc.total_chunks} chunks
                              </span>
                              <span>•</span>
                              <span>{doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'}</span>
                              <span>•</span>
                              <span>{formatBytes(doc.file_size_bytes)}</span>
                            </div>

                            <div className="text-[10px] text-slate-500 mt-1">
                              {formatDate(doc.created_at)}
                            </div>
                          </div>
                        </div>

                        {/* Action buttons */}
                        <div className="flex items-center space-x-1 shrink-0">
                          <button
                            onClick={() => handleDeleteClick(doc.document_id)}
                            className="p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-slate-800/80 transition"
                            title="Delete document"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Footer System Info */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-900/60 text-[11px] text-slate-500 flex items-center justify-between">
        <div className="flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-slate-400" />
          <span>ChromaDB Vector Store</span>
        </div>
        <span>all-MiniLM-L6-v2</span>
      </div>
    </aside>
  );
}
