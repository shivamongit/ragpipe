"use client";

import { useEffect, useRef, useState } from "react";
import {
  Send,
  Loader2,
  AlertCircle,
  Sparkles,
  FileText,
  Database,
  Zap,
  TrendingUp,
} from "lucide-react";
import {
  checkHealth,
  getStats,
  StatsResponse,
  createConversation,
  getConversation,
  ConversationDetail,
  streamQuery,
  Source,
} from "@/lib/api";

import { Sidebar } from "@/components/Sidebar";
import { ModelPicker } from "@/components/ModelPicker";
import { IngestPanel } from "@/components/IngestPanel";
import { SettingsModal } from "@/components/SettingsModal";
import { ChatMessage, UIMessage } from "@/components/ChatMessage";

export default function Home() {
  // ── state ──────────────────────────────────────────────────────────────
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [model, setModel] = useState<{ provider: string; model: string } | null>(null);
  const [topK, setTopK] = useState(5);
  const [stats, setStats] = useState<StatsResponse>({ documents: 0, chunks: 0 });
  const [health, setHealth] = useState<"online" | "offline" | "unknown">("unknown");
  const [showIngest, setShowIngest] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<{ close: () => void } | null>(null);

  // ── effects ────────────────────────────────────────────────────────────

  // Restore last selected model
  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem("ragpipe.model");
    if (saved) {
      try {
        setModel(JSON.parse(saved));
      } catch {}
    }
    const k = window.localStorage.getItem("ragpipe.topK");
    if (k) setTopK(Number(k));
  }, []);

  // Persist model + topK
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (model) window.localStorage.setItem("ragpipe.model", JSON.stringify(model));
  }, [model]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("ragpipe.topK", String(topK));
  }, [topK]);

  // Health + stats polling
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await checkHealth();
        if (!alive) return;
        setHealth(h ? "online" : "offline");
        if (h) {
          try {
            const s = await getStats();
            if (alive) setStats(s);
          } catch {}
        }
      } catch {
        if (alive) setHealth("offline");
      }
    };
    poll();
    const id = setInterval(poll, 15_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Auto-scroll on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // ── handlers ───────────────────────────────────────────────────────────

  const handleNewChat = () => {
    streamRef.current?.close();
    setConversationId(null);
    setMessages([]);
    setError(null);
    inputRef.current?.focus();
  };

  const handleSelectConversation = async (id: string) => {
    if (id === conversationId) return;
    streamRef.current?.close();
    try {
      const conv: ConversationDetail = await getConversation(id);
      setConversationId(id);
      setMessages(
        conv.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.sources,
          model: m.model || undefined,
          tokensUsed: m.tokens_used,
          latencyMs: m.latency_ms,
          timestamp: new Date(m.created_at),
        })),
      );
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load conversation");
    }
  };

  const ensureConversation = async (firstQuestion: string): Promise<string> => {
    if (conversationId) return conversationId;
    const title = firstQuestion.slice(0, 60).trim() || "New chat";
    const conv = await createConversation({
      title,
      provider: model?.provider,
      model: model?.model,
    });
    setConversationId(conv.id);
    setSidebarRefresh((x) => x + 1);
    return conv.id;
  };

  const handleSend = async (questionOverride?: string) => {
    const question = (questionOverride ?? input).trim();
    if (!question || streaming) return;
    setError(null);
    setInput("");

    // Optimistic user message
    const userMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      timestamp: new Date(),
    };
    const assistantMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      isStreaming: true,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    try {
      const convId = await ensureConversation(question);

      let sourcesBuf: Source[] = [];
      let answerBuf = "";
      const handle = streamQuery(question, {
        topK,
        provider: model?.provider,
        model: model?.model,
        conversationId: convId,
        onSources: (s) => {
          sourcesBuf = s;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsg.id ? { ...m, sources: s } : m)),
          );
        },
        onToken: (tok) => {
          answerBuf += tok;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsg.id ? { ...m, content: answerBuf } : m)),
          );
        },
        onDone: ({ model: modelId, latency_ms }) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id
                ? {
                    ...m,
                    content: answerBuf || "(no answer)",
                    isStreaming: false,
                    model: modelId,
                    latencyMs: latency_ms,
                    sources: sourcesBuf,
                  }
                : m,
            ),
          );
          setSidebarRefresh((x) => x + 1);
        },
        onError: (msg) => {
          setError(msg);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id
                ? { ...m, content: `_Error: ${msg}_`, isStreaming: false }
                : m,
            ),
          );
        },
      });
      streamRef.current = handle;
      await handle.promise;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Query failed";
      setError(msg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? { ...m, content: `_Error: ${msg}_`, isStreaming: false }
            : m,
        ),
      );
    } finally {
      setStreaming(false);
      streamRef.current = null;
    }
  };

  const handleStopGeneration = () => {
    streamRef.current?.close();
    streamRef.current = null;
    setStreaming(false);
    setMessages((prev) => prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m)));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleIngested = async () => {
    try {
      const s = await getStats();
      setStats(s);
    } catch {}
  };

  // ── render ─────────────────────────────────────────────────────────────

  return (
    <div className="h-full flex relative">
      <Sidebar
        activeId={conversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
        onOpenSettings={() => setShowSettings(true)}
        onOpenIngest={() => setShowIngest(true)}
        stats={stats}
        health={health}
        refreshKey={sidebarRefresh}
      />

      <main className="flex-1 flex flex-col min-w-0 relative z-10">
        {/* Top bar */}
        <header className="flex items-center justify-between px-5 py-3 border-b border-border bg-background/60 backdrop-blur-sm">
          <div className="flex items-center gap-2 min-w-0">
            <h2 className="text-sm font-medium truncate">
              {messages.length === 0 ? (
                <span className="text-foreground-muted">Start a new conversation</span>
              ) : (
                <span>Chat</span>
              )}
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowIngest(true)}
              className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors"
            >
              <Database className="w-3.5 h-3.5" />
              <span>{stats.documents} docs</span>
            </button>
            <ModelPicker value={model} onChange={setModel} />
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="mx-5 mt-3 px-4 py-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive flex items-center gap-2 animate-slide-up">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button onClick={() => setError(null)} className="text-xs hover:underline">
              dismiss
            </button>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState
              onSuggestion={(q) => handleSend(q)}
              hasDocuments={stats.documents > 0}
              onIngest={() => setShowIngest(true)}
              onSelectModel={() => {}}
              modelSelected={!!model}
            />
          ) : (
            <div className="pb-8">
              {messages.map((m) => (
                <ChatMessage key={m.id} message={m} />
              ))}
              {streaming && messages[messages.length - 1]?.content === "" && (
                <div className="max-w-[850px] mx-auto px-6 py-3 flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <span className="dot-loader">
                    <span /> <span /> <span />
                  </span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="px-5 pb-5 pt-2">
          <div className="max-w-[850px] mx-auto">
            <div className="relative rounded-2xl border border-border bg-surface-2/80 backdrop-blur-sm focus-within:border-border-strong shadow-lg transition-all">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  health === "offline"
                    ? "Backend offline — start the ragpipe server"
                    : stats.documents === 0
                    ? "Ingest documents first, then ask anything..."
                    : "Ask anything about your documents..."
                }
                rows={1}
                disabled={health === "offline"}
                className="w-full resize-none bg-transparent px-4 py-3.5 pr-12 text-[15px] focus:outline-none placeholder:text-foreground-subtle disabled:opacity-50"
                style={{ minHeight: "52px", maxHeight: "240px" }}
                onInput={(e) => {
                  const t = e.currentTarget;
                  t.style.height = "auto";
                  t.style.height = Math.min(t.scrollHeight, 240) + "px";
                }}
              />
              <div className="absolute right-2 bottom-2">
                {streaming ? (
                  <button
                    onClick={handleStopGeneration}
                    className="w-9 h-9 rounded-xl bg-destructive hover:bg-destructive/90 text-destructive-foreground flex items-center justify-center transition-colors"
                    title="Stop"
                  >
                    <span className="w-2.5 h-2.5 bg-current rounded-sm" />
                  </button>
                ) : (
                  <button
                    onClick={() => handleSend()}
                    disabled={!input.trim() || health === "offline"}
                    className="w-9 h-9 rounded-xl bg-accent hover:bg-accent-hover text-accent-foreground disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors shadow-sm"
                    title="Send (Enter)"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px] text-foreground-subtle px-1">
              <span>
                <kbd className="px-1.5 py-0.5 rounded bg-surface-3 border border-border text-foreground-muted">
                  ⏎
                </kbd>{" "}
                send ·{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-surface-3 border border-border text-foreground-muted">
                  ⇧⏎
                </kbd>{" "}
                newline
              </span>
              <span className="flex items-center gap-1">
                <TrendingUp className="w-3 h-3" /> top_k={topK}
              </span>
            </div>
          </div>
        </div>
      </main>

      <IngestPanel
        open={showIngest}
        onClose={() => setShowIngest(false)}
        stats={stats}
        onIngested={handleIngested}
      />
      <SettingsModal
        open={showSettings}
        onClose={() => setShowSettings(false)}
        topK={topK}
        onTopKChange={setTopK}
      />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────

interface EmptyStateProps {
  onSuggestion: (q: string) => void;
  hasDocuments: boolean;
  onIngest: () => void;
  onSelectModel: () => void;
  modelSelected: boolean;
}

function EmptyState({ onSuggestion, hasDocuments, onIngest }: EmptyStateProps) {
  const suggestions = hasDocuments
    ? [
        { icon: Sparkles, label: "Summarize the main themes" },
        { icon: TrendingUp, label: "What are the key findings?" },
        { icon: FileText, label: "List all recommendations" },
        { icon: Zap, label: "Compare the methodologies" },
      ]
    : [];

  return (
    <div className="h-full flex items-center justify-center px-6">
      <div className="max-w-2xl w-full animate-slide-up">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 mb-5 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-xl shadow-indigo-500/30">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-semibold tracking-tight mb-2">
            <span className="bg-gradient-to-r from-foreground to-foreground-muted bg-clip-text text-transparent">
              Welcome to ragpipe
            </span>
          </h1>
          <p className="text-foreground-muted max-w-md mx-auto">
            Production RAG with multi-LLM support. Upload your docs and chat with{" "}
            <span className="text-foreground">OpenAI, Claude, Gemini, Groq, Cohere, Mistral</span>{" "}
            or local <span className="text-foreground">Ollama</span> models.
          </p>
        </div>

        {!hasDocuments ? (
          <div className="mt-8 rounded-2xl border border-border bg-surface-2/50 p-6 text-center">
            <Database className="w-8 h-8 text-foreground-muted mx-auto mb-3" />
            <h2 className="text-base font-semibold mb-1">No documents yet</h2>
            <p className="text-sm text-foreground-muted mb-4">
              Upload PDFs, DOCX, TXT, or paste text to start chatting with your knowledge base.
            </p>
            <button
              onClick={onIngest}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover text-accent-foreground text-sm font-medium transition-colors shadow-sm"
            >
              <Database className="w-4 h-4" /> Open Knowledge Base
            </button>
          </div>
        ) : (
          <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {suggestions.map((s, i) => (
              <button
                key={i}
                onClick={() => onSuggestion(s.label)}
                className="group flex items-start gap-3 p-3.5 rounded-xl border border-border bg-surface-2/40 hover:bg-surface-2 hover:border-border-strong text-left transition-all"
              >
                <span className="w-8 h-8 rounded-lg bg-accent/10 text-accent-hover flex items-center justify-center shrink-0 group-hover:bg-accent/20 transition-colors">
                  <s.icon className="w-4 h-4" />
                </span>
                <span className="text-sm text-foreground-muted group-hover:text-foreground transition-colors">
                  {s.label}
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="mt-8 flex items-center justify-center gap-3 text-xs text-foreground-subtle">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-success pulse-dot" />
            7 providers
          </span>
          <span>·</span>
          <span>30+ models</span>
          <span>·</span>
          <span>Streaming &amp; citations</span>
        </div>
      </div>
    </div>
  );
}
