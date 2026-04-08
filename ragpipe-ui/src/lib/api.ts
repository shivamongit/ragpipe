const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
}

export interface StatsResponse {
  documents: number;
  chunks: number;
}

export interface IngestResponse {
  documents: number;
  chunks: number;
}

export async function queryPipeline(
  question: string,
  apiKey?: string,
  topK?: number
): Promise<QueryResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers,
    body: JSON.stringify({ question, top_k: topK }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Query failed (${res.status}): ${err}`);
  }

  return res.json();
}

export async function streamQuery(
  question: string,
  apiKey?: string,
  topK?: number,
  onToken: (token: string) => void = () => {},
  onDone: () => void = () => {}
): Promise<void> {
  const wsUrl = API_BASE.replace(/^http/, "ws") + "/query/stream";
  const ws = new WebSocket(wsUrl);

  return new Promise((resolve, reject) => {
    ws.onopen = () => {
      if (apiKey) {
        ws.send(JSON.stringify({ api_key: apiKey }));
      }
      ws.send(JSON.stringify({ question, top_k: topK }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "token") {
        onToken(data.content);
      } else if (data.type === "done") {
        onDone();
        ws.close();
        resolve();
      }
    };

    ws.onerror = () => {
      reject(new Error("WebSocket connection failed"));
    };

    ws.onclose = (event) => {
      if (event.code !== 1000 && event.code !== 1005) {
        reject(new Error(`WebSocket closed: ${event.reason || event.code}`));
      }
    };
  });
}

export async function ingestDocuments(
  documents: { content: string; metadata?: Record<string, string> }[],
  apiKey?: string
): Promise<IngestResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers,
    body: JSON.stringify({ documents }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ingest failed (${res.status}): ${err}`);
  }

  return res.json();
}

export async function getStats(apiKey?: string): Promise<StatsResponse> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${API_BASE}/stats`, { headers });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Stats failed (${res.status}): ${err}`);
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
