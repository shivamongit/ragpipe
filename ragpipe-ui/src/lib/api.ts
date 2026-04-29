/**
 * Type-safe API client for the ragpipe FastAPI backend.
 *
 * All requests honor an optional X-API-Key header. The base URL is configurable
 * via NEXT_PUBLIC_API_URL or the runtime override returned by `getApiBase()`.
 */

const DEFAULT_BASE = "http://localhost:8000";

export function getApiBase(): string {
  if (typeof window !== "undefined") {
    const stored = window.localStorage.getItem("ragpipe.apiUrl");
    if (stored) return stored;
  }
  return process.env.NEXT_PUBLIC_API_URL || DEFAULT_BASE;
}

export function getApiKey(): string | undefined {
  if (typeof window === "undefined") return undefined;
  return window.localStorage.getItem("ragpipe.apiKey") || undefined;
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: authHeaders(init.headers as Record<string, string>),
  });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body === "string" ? body : body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`${res.status} ${detail || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Types ───────────────────────────────────────────────────────────────────

export interface Source {
  text: string;
  doc_id: string;
  score: number;
  rank: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  model: string;
  tokens_used: number;
  latency_ms: number;
  conversation_id?: string;
  message_id?: string;
}

export interface StatsResponse {
  documents: number;
  chunks: number;
}

export interface IngestResponse {
  documents: number;
  chunks: number;
}

export interface UploadResponse {
  documents: number;
  chunks: number;
  files: string[];
}

export interface ProviderInfo {
  id: string;
  name: string;
  requires_api_key: boolean;
  api_key_env_var: string;
  docs_url: string;
  available: boolean;
  model_count: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  provider_name: string;
  context_window: number;
  input_cost_per_m: number;
  output_cost_per_m: number;
  streaming: boolean;
  description: string;
  tags: string[];
  available: boolean;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  model?: string;
  tokens_used: number;
  latency_ms: number;
  created_at: number;
}

export interface Conversation {
  id: string;
  title: string;
  model?: string;
  provider?: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

// ── Provider/model introspection ────────────────────────────────────────────

export async function listProviders(): Promise<ProviderInfo[]> {
  const res = await api<{ providers: ProviderInfo[] }>("/providers");
  return res.providers;
}

export async function listModels(opts: { provider?: string; availableOnly?: boolean } = {}): Promise<ModelInfo[]> {
  const params = new URLSearchParams();
  if (opts.provider) params.set("provider", opts.provider);
  if (opts.availableOnly) params.set("available_only", "true");
  const qs = params.toString() ? `?${params.toString()}` : "";
  const res = await api<{ models: ModelInfo[] }>(`/models${qs}`);
  return res.models;
}

// ── Pipeline operations ─────────────────────────────────────────────────────

export interface QueryOptions {
  topK?: number;
  model?: string;
  provider?: string;
  apiKeyOverride?: string;
  conversationId?: string;
}

export async function queryPipeline(question: string, opts: QueryOptions = {}): Promise<QueryResponse> {
  return api<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: opts.topK,
      model: opts.model,
      provider: opts.provider,
      api_key_override: opts.apiKeyOverride,
      conversation_id: opts.conversationId,
    }),
  });
}

export interface StreamCallbacks {
  onSources?: (sources: Source[]) => void;
  onToken?: (token: string) => void;
  onDone?: (info: { model: string; latency_ms: number; conversation_id?: string; message_id?: string }) => void;
  onError?: (msg: string) => void;
}

export function streamQuery(
  question: string,
  opts: QueryOptions & StreamCallbacks = {},
): { close: () => void; promise: Promise<void> } {
  const wsUrl = getApiBase().replace(/^http/, "ws") + "/query/stream";
  const ws = new WebSocket(wsUrl);
  const apiKey = getApiKey();

  const promise = new Promise<void>((resolve, reject) => {
    ws.onopen = () => {
      if (apiKey) ws.send(JSON.stringify({ api_key: apiKey }));
      ws.send(JSON.stringify({
        question,
        top_k: opts.topK,
        model: opts.model,
        provider: opts.provider,
        api_key_override: opts.apiKeyOverride,
        conversation_id: opts.conversationId,
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "sources":
            opts.onSources?.(data.sources);
            break;
          case "token":
            opts.onToken?.(data.content);
            break;
          case "done":
            opts.onDone?.({
              model: data.model,
              latency_ms: data.latency_ms,
              conversation_id: data.conversation_id,
              message_id: data.message_id,
            });
            ws.close();
            resolve();
            break;
          case "error":
            opts.onError?.(data.message);
            ws.close();
            reject(new Error(data.message));
            break;
        }
      } catch (e) {
        // Ignore malformed frames
      }
    };

    ws.onerror = () => reject(new Error("WebSocket connection failed"));
    ws.onclose = (event) => {
      if (event.code !== 1000 && event.code !== 1005) {
        reject(new Error(`WebSocket closed: ${event.reason || event.code}`));
      }
    };
  });

  return { close: () => ws.close(), promise };
}

// ── Ingestion ───────────────────────────────────────────────────────────────

export async function ingestDocuments(
  documents: { content: string; metadata?: Record<string, string> }[],
): Promise<IngestResponse> {
  return api<IngestResponse>("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ documents }),
  });
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const res = await fetch(`${getApiBase()}/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Stats / Health / Index ──────────────────────────────────────────────────

export async function getStats(): Promise<StatsResponse> {
  return api<StatsResponse>("/stats");
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${getApiBase()}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function clearIndex(): Promise<void> {
  await api("/index", { method: "DELETE" });
}

// ── Conversations ───────────────────────────────────────────────────────────

export async function listConversations(): Promise<Conversation[]> {
  const res = await api<{ conversations: Conversation[] }>("/conversations");
  return res.conversations;
}

export async function createConversation(opts: { title?: string; model?: string; provider?: string } = {}): Promise<Conversation> {
  return api<Conversation>("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: opts.title || "New chat",
      model: opts.model,
      provider: opts.provider,
    }),
  });
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  return api<ConversationDetail>(`/conversations/${id}`);
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await api(`/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(id: string): Promise<void> {
  await api(`/conversations/${id}`, { method: "DELETE" });
}

// ── Settings persistence ────────────────────────────────────────────────────

export function setApiBase(url: string): void {
  if (typeof window !== "undefined") {
    if (url) window.localStorage.setItem("ragpipe.apiUrl", url);
    else window.localStorage.removeItem("ragpipe.apiUrl");
  }
}

export function setApiKey(key: string): void {
  if (typeof window !== "undefined") {
    if (key) window.localStorage.setItem("ragpipe.apiKey", key);
    else window.localStorage.removeItem("ragpipe.apiKey");
  }
}

export function getProviderApiKey(provider: string): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(`ragpipe.providerKey.${provider}`) || "";
}

export function setProviderApiKey(provider: string, key: string): void {
  if (typeof window === "undefined") return;
  if (key) {
    window.localStorage.setItem(`ragpipe.providerKey.${provider}`, key);
  } else {
    window.localStorage.removeItem(`ragpipe.providerKey.${provider}`);
  }
}
