"use client";

import { useEffect, useState } from "react";
import { X, Key, Server, Sliders, ExternalLink, Check, Eye, EyeOff } from "lucide-react";
import {
  ProviderInfo,
  listProviders,
  setApiBase,
  getApiBase,
  setApiKey,
  getApiKey,
  setProviderApiKey,
  getProviderApiKey,
} from "@/lib/api";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  topK: number;
  onTopKChange: (n: number) => void;
}

export function SettingsModal({ open, onClose, topK, onTopKChange }: SettingsModalProps) {
  const [tab, setTab] = useState<"providers" | "server" | "retrieval">("providers");
  const [apiUrl, setApiUrl] = useState("");
  const [apiKeyVal, setApiKeyVal] = useState("");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [providerKeys, setProviderKeys] = useState<Record<string, string>>({});
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    if (!open) return;
    setApiUrl(getApiBase());
    setApiKeyVal(getApiKey() || "");
    listProviders()
      .then((p) => {
        setProviders(p);
        const keys: Record<string, string> = {};
        for (const prov of p) {
          if (prov.requires_api_key) {
            keys[prov.id] = getProviderApiKey(prov.id);
          }
        }
        setProviderKeys(keys);
      })
      .catch(() => setProviders([]));
  }, [open]);

  if (!open) return null;

  const handleSave = () => {
    setApiBase(apiUrl);
    setApiKey(apiKeyVal);
    for (const [pid, key] of Object.entries(providerKeys)) {
      setProviderApiKey(pid, key);
    }
    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1500);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
      style={{ background: "color-mix(in srgb, var(--background) 70%, transparent)" }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-3xl max-h-[85vh] bg-surface border border-border rounded-2xl shadow-2xl flex overflow-hidden animate-slide-up"
      >
        {/* Sidebar tabs */}
        <div className="w-44 shrink-0 border-r border-border bg-surface-2/40 p-3">
          <div className="text-[10px] uppercase tracking-wider text-foreground-subtle font-medium mb-2 px-2">
            Settings
          </div>
          <SettingsTab active={tab === "providers"} onClick={() => setTab("providers")}>
            <Key className="w-3.5 h-3.5" /> Providers
          </SettingsTab>
          <SettingsTab active={tab === "server"} onClick={() => setTab("server")}>
            <Server className="w-3.5 h-3.5" /> Server
          </SettingsTab>
          <SettingsTab active={tab === "retrieval"} onClick={() => setTab("retrieval")}>
            <Sliders className="w-3.5 h-3.5" /> Retrieval
          </SettingsTab>
        </div>

        {/* Main */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center justify-between px-5 py-4 border-b border-border">
            <h2 className="text-base font-semibold capitalize">
              {tab === "providers" && "LLM Providers"}
              {tab === "server" && "Server Connection"}
              {tab === "retrieval" && "Retrieval Settings"}
            </h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-surface-3 text-foreground-muted hover:text-foreground"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-4">
            {tab === "providers" && (
              <div className="space-y-3">
                <p className="text-xs text-foreground-muted">
                  API keys are stored locally in your browser and forwarded to the
                  backend per-request. They never leave your machine without your action.
                </p>
                {providers
                  .filter((p) => p.requires_api_key)
                  .map((p) => (
                    <div
                      key={p.id}
                      className="rounded-lg border border-border bg-surface-2/50 p-3"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">{p.name}</span>
                          {p.available && (
                            <span className="flex items-center gap-1 text-[10px] text-success">
                              <Check className="w-3 h-3" /> Available
                            </span>
                          )}
                          {!p.available && providerKeys[p.id] && (
                            <span className="text-[10px] text-warning">
                              Key set, but package not installed
                            </span>
                          )}
                        </div>
                        <a
                          href={p.docs_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[11px] text-foreground-muted hover:text-foreground flex items-center gap-1"
                        >
                          docs <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          type={showKeys[p.id] ? "text" : "password"}
                          value={providerKeys[p.id] || ""}
                          onChange={(e) =>
                            setProviderKeys((prev) => ({ ...prev, [p.id]: e.target.value }))
                          }
                          placeholder={`${p.api_key_env_var} (e.g. sk-...)`}
                          className="flex-1 bg-surface-3 border border-border rounded-md px-3 py-1.5 text-sm font-mono focus:outline-none focus:border-accent"
                          autoComplete="off"
                          spellCheck={false}
                        />
                        <button
                          onClick={() =>
                            setShowKeys((prev) => ({ ...prev, [p.id]: !prev[p.id] }))
                          }
                          className="p-1.5 rounded-md hover:bg-surface-3 text-foreground-muted"
                        >
                          {showKeys[p.id] ? (
                            <EyeOff className="w-3.5 h-3.5" />
                          ) : (
                            <Eye className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                <div className="rounded-lg border border-border bg-surface-2/50 p-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Ollama (Local)</span>
                    <span className="text-[10px] text-success flex items-center gap-1">
                      <Check className="w-3 h-3" /> No API key needed
                    </span>
                  </div>
                  <p className="text-xs text-foreground-muted mt-1">
                    Runs locally on your machine. Install at{" "}
                    <a
                      href="https://ollama.com"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:text-foreground"
                    >
                      ollama.com
                    </a>
                  </p>
                </div>
              </div>
            )}

            {tab === "server" && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Backend URL
                  </label>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="http://localhost:8000"
                    className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent"
                  />
                  <p className="text-xs text-foreground-muted mt-1.5">
                    The ragpipe FastAPI server. Default is http://localhost:8000.
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Server API Key (optional)
                  </label>
                  <input
                    type="password"
                    value={apiKeyVal}
                    onChange={(e) => setApiKeyVal(e.target.value)}
                    placeholder="Only if your server has --api-key set"
                    className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent"
                    autoComplete="off"
                  />
                  <p className="text-xs text-foreground-muted mt-1.5">
                    Sent as <code className="text-[11px]">X-API-Key</code> header.
                  </p>
                </div>
              </div>
            )}

            {tab === "retrieval" && (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium">Top K</label>
                    <span className="text-sm tabular-nums text-foreground">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(e) => onTopKChange(Number(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                  <p className="text-xs text-foreground-muted mt-1.5">
                    Number of chunks retrieved per query. More chunks = more context but
                    higher latency and token cost.
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-end gap-3 px-5 py-3 border-t border-border bg-surface-2/30">
            {savedFlash && (
              <span className="text-xs text-success flex items-center gap-1 animate-fade-in">
                <Check className="w-3.5 h-3.5" /> Saved
              </span>
            )}
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg text-sm text-foreground-muted hover:bg-surface-3 hover:text-foreground transition-colors"
            >
              Close
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 rounded-lg bg-accent hover:bg-accent-hover text-accent-foreground text-sm font-medium transition-colors shadow-sm"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingsTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-md text-sm transition-colors ${
        active
          ? "bg-surface-3 text-foreground"
          : "text-foreground-muted hover:text-foreground hover:bg-surface-3/50"
      }`}
    >
      {children}
    </button>
  );
}
