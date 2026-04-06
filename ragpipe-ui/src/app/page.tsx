"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Upload,
  Database,
  Settings,
  X,
  FileText,
  Clock,
  Cpu,
  Zap,
  ChevronDown,
  Circle,
  BookOpen,
  Layers,
} from "lucide-react";
import { queryPipeline, ingestDocuments, getStats, checkHealth } from "@/lib/api";
import type { Source, StatsResponse } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  model?: string;
  tokensUsed?: number;
  latencyMs?: number;
  timestamp: Date;
}

function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <button
      onClick={() => setExpanded(!expanded)}
      className="text-left w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-3 hover:border-[#404040] transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono bg-accent/20 text-accent px-1.5 py-0.5 rounded">
            [{index + 1}]
          </span>
          <span className="text-xs text-muted-foreground truncate max-w-[200px]">
            {source.doc_id || "Unknown source"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {(source.score * 100).toFixed(0)}%
          </span>
          <ChevronDown
            className={`w-3 h-3 text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </div>
      </div>
      {expanded && (
        <p className="mt-2 text-xs text-muted-foreground leading-relaxed border-t border-[#2a2a2a] pt-2">
          {source.text}
        </p>
      )}
    </button>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`animate-fade-in ${isUser ? "flex justify-end" : ""}`}>
      <div className={`max-w-[720px] ${isUser ? "ml-16" : "mr-16"}`}>
        {/* Message bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "bg-accent text-accent-foreground"
              : "bg-[#161616]"
          }`}
        >
          <div className="prose-chat text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
        </div>

        {/* Metadata bar for assistant messages */}
        {!isUser && message.model && (
          <div className="flex items-center gap-3 mt-2 px-1">
            {message.model && (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <Cpu className="w-3 h-3" />
                {message.model}
              </span>
            )}
            {message.tokensUsed !== undefined && message.tokensUsed > 0 && (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <Layers className="w-3 h-3" />
                {message.tokensUsed} tokens
              </span>
            )}
            {message.latencyMs !== undefined && (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="w-3 h-3" />
                {formatLatency(message.latencyMs)}
              </span>
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground px-1">
              <BookOpen className="w-3 h-3" />
              {message.sources.length} source{message.sources.length !== 1 ? "s" : ""}
            </span>
            <div className="grid gap-2">
              {message.sources.map((source, i) => (
                <SourceCard key={i} source={source} index={i} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({ onSuggestionClick }: { onSuggestionClick: (q: string) => void }) {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center animate-slide-up">
        <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-[#161616] border border-[#2a2a2a] flex items-center justify-center">
          <Zap className="w-8 h-8 text-accent" />
        </div>
        <h1 className="text-2xl font-semibold mb-2">ragpipe</h1>
        <p className="text-muted-foreground text-sm max-w-md mb-8">
          Context Engineering Platform. Chat with your documents using
          knowledge graphs, agentic retrieval, and self-improving pipelines.
        </p>
        <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto">
          {[
            "What are the key findings?",
            "Summarize the main topics",
            "Compare the methodologies",
            "List all recommendations",
          ].map((q, i) => (
            <button
              key={i}
              onClick={() => onSuggestionClick(q)}
              className="text-left text-xs bg-[#161616] border border-[#2a2a2a] rounded-xl px-3 py-2.5 text-muted-foreground hover:text-foreground hover:border-[#404040] transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function IngestModal({
  onClose,
  onIngest,
}: {
  onClose: () => void;
  onIngest: (docs: { content: string; metadata?: Record<string, string> }[]) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [textInput, setTextInput] = useState("");
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter(
      (f) => f.type.startsWith("text/") || f.name.endsWith(".md") || f.name.endsWith(".txt")
    );
    setFiles((prev) => [...prev, ...dropped]);
  }, []);

  const handleSubmit = async () => {
    setLoading(true);
    const docs: { content: string; metadata?: Record<string, string> }[] = [];

    for (const file of files) {
      const content = await file.text();
      docs.push({ content, metadata: { source: file.name, type: file.type } });
    }

    if (textInput.trim()) {
      docs.push({ content: textInput.trim(), metadata: { source: "manual_input" } });
    }

    if (docs.length > 0) {
      onIngest(docs);
    }
    setLoading(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#111] border border-[#2a2a2a] rounded-2xl w-full max-w-lg mx-4 animate-slide-up">
        <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
          <h2 className="font-semibold">Ingest Documents</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {/* Drop zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-[#2a2a2a] rounded-xl p-8 text-center cursor-pointer hover:border-accent/50 transition-colors"
          >
            <Upload className="w-8 h-8 mx-auto mb-3 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Drop files here or <span className="text-accent">browse</span>
            </p>
            <p className="text-xs text-muted-foreground mt-1">.txt, .md files</p>
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".txt,.md,.csv"
              className="hidden"
              onChange={(e) => {
                if (e.target.files) setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
              }}
            />
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-2">
              {files.map((f, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-[#1a1a1a] rounded-lg px-3 py-2"
                >
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm truncate max-w-[300px]">{f.name}</span>
                  </div>
                  <button
                    onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Text input */}
          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Or paste text content here..."
            rows={4}
            className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl px-3 py-2 text-sm resize-none focus:outline-none focus:border-accent/50 placeholder-[#555]"
          />
        </div>

        <div className="flex justify-end gap-3 p-4 border-t border-[#2a2a2a]">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading || (files.length === 0 && !textInput.trim())}
            className="px-4 py-2 text-sm bg-accent text-accent-foreground rounded-lg font-medium disabled:opacity-40 hover:bg-accent/90 transition-colors"
          >
            {loading ? "Ingesting..." : "Ingest"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SettingsModal({
  onClose,
  apiKey,
  setApiKey,
  apiUrl,
  setApiUrl,
  topK,
  setTopK,
}: {
  onClose: () => void;
  apiKey: string;
  setApiKey: (v: string) => void;
  apiUrl: string;
  setApiUrl: (v: string) => void;
  topK: number;
  setTopK: (v: number) => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#111] border border-[#2a2a2a] rounded-2xl w-full max-w-md mx-4 animate-slide-up">
        <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
          <h2 className="font-semibold">Settings</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">
              API URL
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent/50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">
              API Key (optional)
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Leave empty if not required"
              className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent/50 placeholder-[#555]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">
              Top K Results
            </label>
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent/50"
            />
          </div>
        </div>

        <div className="flex justify-end p-4 border-t border-[#2a2a2a]">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-accent text-accent-foreground rounded-lg font-medium hover:bg-accent/90 transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showIngest, setShowIngest] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [topK, setTopK] = useState(5);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [notification, setNotification] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check health on mount and periodically
  useEffect(() => {
    const check = async () => {
      const online = await checkHealth();
      setServerOnline(online);
      if (online) {
        try {
          const s = await getStats(apiKey || undefined);
          setStats(s);
        } catch {
          /* ignore */
        }
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, [apiKey]);

  // Auto-clear notification
  useEffect(() => {
    if (notification) {
      const t = setTimeout(() => setNotification(null), 4000);
      return () => clearTimeout(t);
    }
  }, [notification]);

  const showNotification = (type: "success" | "error", message: string) => {
    setNotification({ type, message });
  };

  const handleSuggestionClick = (q: string) => {
    setInput(q);
    // Use a small delay so the input state updates before sending
    setTimeout(() => {
      const syntheticQuestion = q;
      sendQuestion(syntheticQuestion);
    }, 50);
  };

  const sendQuestion = async (question: string) => {
    if (!question || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const result = await queryPipeline(question, apiKey || undefined, topK);
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer,
        sources: result.sources,
        model: result.model,
        tokensUsed: result.tokens_used,
        latencyMs: result.latency_ms,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `**Error:** ${err instanceof Error ? err.message : "Failed to query pipeline"}. Make sure the ragpipe server is running (\`python -m ragpipe serve\`).`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleIngest = async (
    docs: { content: string; metadata?: Record<string, string> }[]
  ) => {
    try {
      const result = await ingestDocuments(docs, apiKey || undefined);
      showNotification("success", `Ingested ${result.documents} docs → ${result.chunks} chunks`);
      const s = await getStats(apiKey || undefined);
      setStats(s);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Ingest failed");
    }
  };

  const handleSend = () => {
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    sendQuestion(question);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Top bar */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-[#0d0d0d]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
            <Zap className="w-4 h-4 text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-none">ragpipe</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Context Engineering Platform</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Server status */}
          <div className="flex items-center gap-1.5 mr-2">
            <Circle
              className={`w-2 h-2 fill-current ${
                serverOnline === null
                  ? "text-yellow-500"
                  : serverOnline
                    ? "text-success"
                    : "text-destructive"
              }`}
            />
            <span className="text-xs text-muted-foreground">
              {serverOnline === null ? "Checking..." : serverOnline ? "Online" : "Offline"}
            </span>
          </div>

          {/* Stats badge */}
          {stats && (
            <div className="flex items-center gap-1.5 bg-[#161616] border border-[#2a2a2a] rounded-lg px-2.5 py-1.5 mr-1">
              <Database className="w-3 h-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                {stats.documents} docs · {stats.chunks} chunks
              </span>
            </div>
          )}

          <button
            onClick={() => setShowIngest(true)}
            className="flex items-center gap-1.5 bg-[#161616] border border-[#2a2a2a] rounded-lg px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:border-[#404040] transition-colors"
          >
            <Upload className="w-3.5 h-3.5" />
            Ingest
          </button>
          <button
            onClick={() => setShowSettings(true)}
            className="flex items-center justify-center w-8 h-8 rounded-lg border border-[#2a2a2a] text-muted-foreground hover:text-foreground hover:border-[#404040] transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Notification toast */}
      {notification && (
        <div
          className={`mx-4 mt-3 px-4 py-2.5 rounded-xl text-sm animate-slide-up border ${
            notification.type === "success"
              ? "bg-success/10 border-success/20 text-success"
              : "bg-destructive/10 border-destructive/20 text-destructive"
          }`}
        >
          {notification.message}
        </div>
      )}

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState onSuggestionClick={handleSuggestionClick} />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}

            {/* Loading indicator */}
            {loading && (
              <div className="animate-fade-in">
                <div className="max-w-[720px] mr-16">
                  <div className="rounded-2xl px-4 py-3 bg-[#161616]">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-accent/60 rounded-full animate-bounce [animation-delay:0ms]" />
                        <span className="w-2 h-2 bg-accent/60 rounded-full animate-bounce [animation-delay:150ms]" />
                        <span className="w-2 h-2 bg-accent/60 rounded-full animate-bounce [animation-delay:300ms]" />
                      </div>
                      <span className="text-xs text-muted-foreground">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-border bg-[#0d0d0d] p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-3 bg-[#161616] border border-[#2a2a2a] rounded-2xl px-4 py-3 focus-within:border-accent/40 transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your documents..."
              rows={1}
              className="flex-1 bg-transparent resize-none text-sm focus:outline-none placeholder-[#555] max-h-32 min-h-[20px]"
              style={{ height: "auto" }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "auto";
                target.style.height = Math.min(target.scrollHeight, 128) + "px";
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent text-accent-foreground disabled:opacity-30 hover:bg-accent/90 transition-colors shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-center text-xs text-muted-foreground mt-2">
            ragpipe v3.0 · Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* Modals */}
      {showIngest && <IngestModal onClose={() => setShowIngest(false)} onIngest={handleIngest} />}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          apiKey={apiKey}
          setApiKey={setApiKey}
          apiUrl={apiUrl}
          setApiUrl={setApiUrl}
          topK={topK}
          setTopK={setTopK}
        />
      )}
    </div>
  );
}
