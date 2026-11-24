import React, { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatWorkspace from './components/ChatWorkspace';
import CitationDrawer from './components/CitationDrawer';
import RetrievalInspector from './components/RetrievalInspector';
import { 
  fetchHealth, 
  fetchDocuments, 
  uploadDocument, 
  deleteDocument, 
  askQuestion,
  askQuestionStream,
  createConversation,
  fetchConversations,
  fetchConversation,
  deleteConversation,
} from './api/client';
import { Menu, X } from 'lucide-react';

export default function App() {
  // Knowledge Base State
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [backendLive, setBackendLive] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState([]);

  // Chat & Retrieval State
  const [messages, setMessages] = useState([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState(null);
  const [retrievalMode, setRetrievalMode] = useState('hybrid');
  const [streamingStatus, setStreamingStatus] = useState(null);

  // Multi-Turn Conversations State
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [loadingConversations, setLoadingConversations] = useState(false);

  // Citation Drawer State
  const [activeCitation, setActiveCitation] = useState(null);
  const [isCitationDrawerOpen, setIsCitationDrawerOpen] = useState(false);

  // Retrieval Inspector State
  const [isRetrievalInspectorOpen, setIsRetrievalInspectorOpen] = useState(false);

  // Mobile sidebar toggle
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // AbortController ref for in-flight streaming
  const abortControllerRef = useRef(null);

  // Probe Backend Health
  const checkHealth = useCallback(async () => {
    try {
      await fetchHealth();
      setBackendLive(true);
    } catch {
      setBackendLive(false);
    }
  }, []);

  // Load Documents from Backend
  const loadDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const res = await fetchDocuments();
      setDocuments(res.documents || []);
      setBackendLive(true);
    } catch (err) {
      console.warn('Failed to load documents:', err.message);
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  // Load Persistent Conversations
  const loadConversations = useCallback(async () => {
    setLoadingConversations(true);
    try {
      const res = await fetchConversations();
      setConversations(res || []);
    } catch (err) {
      console.warn('Failed to load conversations:', err.message);
    } finally {
      setLoadingConversations(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    checkHealth();
    loadDocuments();
    loadConversations();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth, loadDocuments, loadConversations]);

  // Select active conversation and load message history
  const handleSelectConversation = async (convId) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setQueryLoading(false);
    setStreamingStatus(null);
    setQueryError(null);
    setCurrentConversationId(convId);

    try {
      const detail = await fetchConversation(convId);
      const loadedMessages = (detail.messages || []).map((msg) => {
        const metadata = msg.metadata || {};
        return {
          id: msg.id,
          role: msg.role,
          content: msg.content,
          raw_query: msg.raw_query,
          rewritten_query: msg.rewritten_query,
          citations: metadata.citations || [],
          insufficient_context: Boolean(metadata.insufficient_context),
          isVerified: msg.role === 'assistant',
          isStreaming: false,
          metadata: metadata,
          timestamp: msg.created_at,
        };
      });
      setMessages(loadedMessages);
    } catch (err) {
      setQueryError(`Failed to load conversation history: ${err.message}`);
    }
  };

  // Start a fresh chat session
  const handleNewChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setQueryLoading(false);
    setStreamingStatus(null);
    setQueryError(null);
    setCurrentConversationId(null);
    setMessages([]);
  };

  // Delete a conversation
  const handleDeleteConversation = async (convId) => {
    try {
      await deleteConversation(convId);
      if (currentConversationId === convId) {
        handleNewChat();
      }
      await loadConversations();
    } catch (err) {
      setQueryError(`Failed to delete conversation: ${err.message}`);
    }
  };

  // Handle Document Upload
  const handleUpload = async (file) => {
    setUploading(true);
    try {
      const res = await uploadDocument(file);
      await loadDocuments();
      return res;
    } finally {
      setUploading(false);
    }
  };

  // Handle Document Delete
  const handleDelete = async (docId) => {
    await deleteDocument(docId);
    // If the deleted document was in selectedDocIds, remove it
    setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
    await loadDocuments();
  };

  // Handle Question Submission via SSE Real-Time Streaming
  const handleSubmitQuestion = async (queryText) => {
    setQueryError(null);
    setStreamingStatus('Searching the knowledge base...');

    // Abort any previous pending stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Auto-create conversation if not currently in an active session
    let activeConvId = currentConversationId;
    if (!activeConvId) {
      try {
        const created = await createConversation({ title: queryText.slice(0, 40) });
        activeConvId = created.id;
        setCurrentConversationId(created.id);
      } catch (convErr) {
        console.warn('Failed to auto-create conversation:', convErr);
      }
    }

    const userMsg = {
      role: 'user',
      content: queryText,
      timestamp: new Date().toISOString(),
    };

    const assistantMsgId = `assistant_${Date.now()}`;
    const initialAssistantMsg = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      isVerified: false,
      statusStage: 'retrieving',
      statusMessage: 'Searching the knowledge base...',
      citations: [],
      insufficient_context: false,
      sources_used: [],
      metadata: {},
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setQueryLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const docFilter = selectedDocIds.length > 0 ? selectedDocIds : null;

    try {
      await askQuestionStream({
        query: queryText,
        retrieval_mode: retrievalMode,
        top_k: 5,
        document_ids: docFilter,
        conversation_id: activeConvId,
        signal: controller.signal,
        onStatus: (statusPayload) => {
          setStreamingStatus(statusPayload.message);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    statusStage: statusPayload.stage,
                    statusMessage: statusPayload.message,
                  }
                : msg
            )
          );
        },
        onToken: (tokenDelta) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    content: msg.content + tokenDelta,
                  }
                : msg
            )
          );
        },
        onCitations: (citationsPayload) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    isVerified: true,
                    isStreaming: false,
                    citations: citationsPayload.citations || [],
                    insufficient_context: Boolean(citationsPayload.insufficient_context),
                    metadata: citationsPayload.metadata || {},
                    ttft_ms: citationsPayload.ttft_ms,
                  }
                : msg
            )
          );
        },
        onRejection: (rejectionPayload) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    content:
                      rejectionPayload.safe_fallback ||
                      'The generated answer could not be verified because it referenced invalid, fabricated, or excluded source citations. The available documents do not substantiate this response.',
                    isVerified: false,
                    isRejected: true,
                    isStreaming: false,
                    rejectionReason: rejectionPayload.reason,
                    citations: [],
                  }
                : msg
            )
          );
          loadConversations();
        },
        onError: (errorPayload) => {
          setQueryError(errorPayload.message || 'Error during streaming generation.');
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    isStreaming: false,
                  }
                : msg
            )
          );
        },
        onDone: () => {
          setQueryLoading(false);
          setStreamingStatus(null);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    isStreaming: false,
                  }
                : msg
            )
          );
          loadConversations();
        },
      });
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }
      setQueryError(err.message || 'An error occurred while answering your question.');
    } finally {
      setQueryLoading(false);
      setStreamingStatus(null);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  // Stop Generation / Cancel Stream
  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setQueryLoading(false);
    setStreamingStatus(null);
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === 'assistant' && last.isStreaming) {
        return [
          ...prev.slice(0, -1),
          {
            ...last,
            isStreaming: false,
            isStopped: true,
            statusMessage: 'Generation stopped by user.',
          },
        ];
      }
      return prev;
    });
  };

  // Handle Citation Inspection
  const handleCitationClick = (citation) => {
    setActiveCitation(citation);
    setIsCitationDrawerOpen(true);
  };

  // Toggle selection filter for document
  const handleToggleDocSelection = (docId) => {
    setSelectedDocIds((prev) => 
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans text-slate-100 antialiased">
      {/* Mobile Sidebar Toggle Button */}
      <div className="md:hidden fixed top-3 left-3 z-40">
        <button
          onClick={() => setMobileSidebarOpen(!mobileSidebarOpen)}
          className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-white shadow-lg"
          title="Toggle Navigation"
        >
          {mobileSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Sidebar (Desktop & Mobile Drawer) */}
      <div className={`fixed md:relative inset-y-0 left-0 z-30 transition-transform duration-300 md:translate-x-0 ${
        mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
      }`}>
        <Sidebar
          documents={documents}
          loading={loadingDocs}
          uploading={uploading}
          backendLive={backendLive}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onRefresh={loadDocuments}
          selectedDocIds={selectedDocIds}
          onToggleDocSelection={handleToggleDocSelection}
          conversations={conversations}
          currentConversationId={currentConversationId}
          onSelectConversation={handleSelectConversation}
          onNewChat={handleNewChat}
          onDeleteConversation={handleDeleteConversation}
        />
      </div>

      {/* Mobile Backdrop */}
      {mobileSidebarOpen && (
        <div
          className="fixed inset-0 bg-slate-950/70 z-20 md:hidden backdrop-blur-xs"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      {/* Main Chat Workspace */}
      <ChatWorkspace
        messages={messages}
        loading={queryLoading}
        error={queryError}
        retrievalMode={retrievalMode}
        streamingStatus={streamingStatus}
        onSelectRetrievalMode={setRetrievalMode}
        onSubmitQuestion={handleSubmitQuestion}
        onStopGeneration={handleStopGeneration}
        onClearChat={handleNewChat}
        onCitationClick={handleCitationClick}
        onOpenRetrievalInspector={() => setIsRetrievalInspectorOpen(true)}
        documentCount={documents.length}
      />

      {/* Verifiable Citation Slide-over Drawer */}
      <CitationDrawer
        isOpen={isCitationDrawerOpen}
        citation={activeCitation}
        onClose={() => {
          setIsCitationDrawerOpen(false);
          setActiveCitation(null);
        }}
      />

      {/* Standalone Retrieval Inspector */}
      <RetrievalInspector
        isOpen={isRetrievalInspectorOpen}
        onClose={() => setIsRetrievalInspectorOpen(false)}
        documents={documents}
      />
    </div>
  );
}
