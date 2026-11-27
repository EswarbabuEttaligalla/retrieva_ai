import React, { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatWorkspace from './components/ChatWorkspace';
import CitationDrawer from './components/CitationDrawer';
import RetrievalInspector from './components/RetrievalInspector';
import AuthScreen, { AuthLoadingScreen } from './components/AuthScreen';
import { useAuth } from './context/AuthContext';
import {
  fetchHealth,
  fetchDocuments,
  uploadDocument,
  deleteDocument,
  askQuestionStream,
  createConversation,
  fetchConversations,
  fetchConversation,
  updateConversationTitle,
  deleteConversation,
} from './api/client';
import { Menu, X } from 'lucide-react';

function AuthenticatedWorkspace({ user, onSignOut }) {
  const ownerId = user.id;

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

  // Citation Drawer & Retrieval Inspector State
  const [activeCitation, setActiveCitation] = useState(null);
  const [isCitationDrawerOpen, setIsCitationDrawerOpen] = useState(false);
  const [isRetrievalInspectorOpen, setIsRetrievalInspectorOpen] = useState(false);

  // Mobile sidebar toggle
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Abort & ownership guards preventing stale User A responses from mutating User B state
  const activeOwnerRef = useRef(ownerId);
  const mountedRef = useRef(true);
  const abortControllerRef = useRef(null);
  const docsAbortRef = useRef(null);
  const convsAbortRef = useRef(null);
  const detailAbortRef = useRef(null);
  const uploadAbortRef = useRef(null);

  const isCurrentSession = useCallback(
    (expectedOwnerId) => mountedRef.current && activeOwnerRef.current === expectedOwnerId,
    []
  );

  const abortAllInFlightRequests = useCallback(() => {
    for (const ref of [
      abortControllerRef,
      docsAbortRef,
      convsAbortRef,
      detailAbortRef,
      uploadAbortRef,
    ]) {
      if (ref.current) {
        ref.current.abort();
        ref.current = null;
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    activeOwnerRef.current = ownerId;
    return () => {
      mountedRef.current = false;
      abortAllInFlightRequests();
    };
  }, [ownerId, abortAllInFlightRequests]);

  // Probe Backend Health
  const checkHealth = useCallback(async () => {
    const requestOwnerId = ownerId;
    try {
      await fetchHealth();
      if (isCurrentSession(requestOwnerId)) {
        setBackendLive(true);
      }
    } catch {
      if (isCurrentSession(requestOwnerId)) {
        setBackendLive(false);
      }
    }
  }, [ownerId, isCurrentSession]);

  // Load Documents for Authenticated User
  const loadDocuments = useCallback(async () => {
    const requestOwnerId = ownerId;
    if (docsAbortRef.current) {
      docsAbortRef.current.abort();
    }
    const controller = new AbortController();
    docsAbortRef.current = controller;

    setLoadingDocs(true);
    try {
      const res = await fetchDocuments({ signal: controller.signal });
      if (!isCurrentSession(requestOwnerId) || controller.signal.aborted) return;
      setDocuments(res.documents || []);
      setBackendLive(true);
    } catch (err) {
      if (err?.name === 'AbortError' || controller.signal.aborted) return;
      if (isCurrentSession(requestOwnerId)) {
        console.warn('Failed to load documents:', err.message);
      }
    } finally {
      if (docsAbortRef.current === controller) {
        docsAbortRef.current = null;
      }
      if (isCurrentSession(requestOwnerId)) {
        setLoadingDocs(false);
      }
    }
  }, [ownerId, isCurrentSession]);

  // Load Persistent Conversations for Authenticated User
  const loadConversations = useCallback(async () => {
    const requestOwnerId = ownerId;
    if (convsAbortRef.current) {
      convsAbortRef.current.abort();
    }
    const controller = new AbortController();
    convsAbortRef.current = controller;

    try {
      const res = await fetchConversations(50, 0, { signal: controller.signal });
      if (!isCurrentSession(requestOwnerId) || controller.signal.aborted) return;
      setConversations(res || []);
    } catch (err) {
      if (err?.name === 'AbortError' || controller.signal.aborted) return;
      if (isCurrentSession(requestOwnerId)) {
        console.warn('Failed to load conversations:', err.message);
      }
    } finally {
      if (convsAbortRef.current === controller) {
        convsAbortRef.current = null;
      }
    }
  }, [ownerId, isCurrentSession]);

  // Initial authenticated load
  useEffect(() => {
    checkHealth();
    loadDocuments();
    loadConversations();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth, loadDocuments, loadConversations]);

  // Explicit Logout with complete state teardown
  const handleSignOut = async () => {
    abortAllInFlightRequests();
    setDocuments([]);
    setSelectedDocIds([]);
    setConversations([]);
    setCurrentConversationId(null);
    setMessages([]);
    setActiveCitation(null);
    setIsCitationDrawerOpen(false);
    setIsRetrievalInspectorOpen(false);
    setQueryError(null);
    setStreamingStatus(null);
    setQueryLoading(false);
    setUploading(false);
    await onSignOut();
  };

  // Select active conversation and load message history
  const handleSelectConversation = async (convId) => {
    const requestOwnerId = ownerId;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (detailAbortRef.current) {
      detailAbortRef.current.abort();
    }
    const controller = new AbortController();
    detailAbortRef.current = controller;

    setQueryLoading(false);
    setStreamingStatus(null);
    setQueryError(null);
    setCurrentConversationId(convId);

    try {
      const detail = await fetchConversation(convId, { signal: controller.signal });
      if (!isCurrentSession(requestOwnerId) || controller.signal.aborted) return;

      const loadedMessages = (detail.messages || []).map((msg) => {
        const metadata = msg.metadata || {};
        const resolvedCitations =
          Array.isArray(msg.citations) && msg.citations.length > 0
            ? msg.citations
            : metadata.citations || [];
        return {
          id: msg.id,
          role: msg.role,
          content: msg.content,
          raw_query: msg.raw_query,
          rewritten_query: msg.rewritten_query,
          citations: resolvedCitations,
          insufficient_context: Boolean(metadata.insufficient_context),
          isVerified: msg.role === 'assistant',
          isStreaming: false,
          metadata,
          timestamp: msg.created_at,
        };
      });
      setMessages(loadedMessages);
    } catch (err) {
      if (err?.name === 'AbortError' || controller.signal.aborted) return;
      if (isCurrentSession(requestOwnerId)) {
        setQueryError(`Failed to load conversation history: ${err.message}`);
      }
    } finally {
      if (detailAbortRef.current === controller) {
        detailAbortRef.current = null;
      }
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

  // Rename an owned conversation
  const handleRenameConversation = async (convId, newTitle) => {
    const requestOwnerId = ownerId;
    await updateConversationTitle(convId, newTitle);
    if (isCurrentSession(requestOwnerId)) {
      await loadConversations();
    }
  };

  // Delete an owned conversation
  const handleDeleteConversation = async (convId) => {
    const requestOwnerId = ownerId;
    try {
      await deleteConversation(convId);
      if (!isCurrentSession(requestOwnerId)) return;
      if (currentConversationId === convId) {
        handleNewChat();
      }
      await loadConversations();
    } catch (err) {
      if (isCurrentSession(requestOwnerId)) {
        setQueryError(`Failed to delete conversation: ${err.message}`);
      }
    }
  };

  // Handle Document Upload
  const handleUpload = async (file) => {
    const requestOwnerId = ownerId;
    if (uploadAbortRef.current) {
      uploadAbortRef.current.abort();
    }
    const controller = new AbortController();
    uploadAbortRef.current = controller;

    setUploading(true);
    try {
      const res = await uploadDocument(file, { signal: controller.signal });
      if (!isCurrentSession(requestOwnerId) || controller.signal.aborted) {
        return res;
      }
      await loadDocuments();
      return res;
    } finally {
      if (uploadAbortRef.current === controller) {
        uploadAbortRef.current = null;
      }
      if (isCurrentSession(requestOwnerId)) {
        setUploading(false);
      }
    }
  };

  // Handle Document Delete
  const handleDelete = async (docId) => {
    const requestOwnerId = ownerId;
    await deleteDocument(docId);
    if (!isCurrentSession(requestOwnerId)) return;
    setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
    await loadDocuments();
  };

  // Handle Question Submission via SSE Real-Time Streaming
  const handleSubmitQuestion = async (queryText) => {
    const requestOwnerId = ownerId;
    setQueryError(null);
    setStreamingStatus('Searching the knowledge base...');

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    let activeConvId = currentConversationId;
    if (!activeConvId) {
      try {
        const created = await createConversation({ title: queryText.slice(0, 40) });
        if (!isCurrentSession(requestOwnerId)) return;
        activeConvId = created.id;
        setCurrentConversationId(created.id);
      } catch (convErr) {
        if (!isCurrentSession(requestOwnerId)) return;
        setStreamingStatus(null);
        setQueryError(convErr.message || 'Failed to start conversation.');
        return;
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
        top_k: 7,
        document_ids: docFilter,
        conversation_id: activeConvId,
        signal: controller.signal,
        onStatus: (statusPayload) => {
          if (!isCurrentSession(requestOwnerId)) return;
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
          if (!isCurrentSession(requestOwnerId)) return;
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
          if (!isCurrentSession(requestOwnerId)) return;
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
          if (!isCurrentSession(requestOwnerId)) return;
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
          if (!isCurrentSession(requestOwnerId)) return;
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
          if (!isCurrentSession(requestOwnerId)) return;
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
      if (controller.signal.aborted || !isCurrentSession(requestOwnerId)) {
        return;
      }
      setQueryError(err.message || 'An error occurred while answering your question.');
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      if (isCurrentSession(requestOwnerId)) {
        setQueryLoading(false);
        setStreamingStatus(null);
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
      <div
        className={`fixed md:relative inset-y-0 left-0 z-30 transition-transform duration-300 md:translate-x-0 ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
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
          onRenameConversation={handleRenameConversation}
          userEmail={user.email}
          onSignOut={handleSignOut}
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
        key={ownerId}
        isOpen={isRetrievalInspectorOpen}
        onClose={() => setIsRetrievalInspectorOpen(false)}
        documents={documents}
      />
    </div>
  );
}

export default function App() {
  const { user, loading, authError, clearAuthError, isConfigured, signIn, signUp, signOut } =
    useAuth();

  if (loading) {
    return <AuthLoadingScreen />;
  }

  if (!user) {
    return (
      <AuthScreen
        onSignIn={signIn}
        onSignUp={signUp}
        authError={authError}
        onClearError={clearAuthError}
        isConfigured={isConfigured}
      />
    );
  }

  return <AuthenticatedWorkspace key={user.id} user={user} onSignOut={signOut} />;
}
