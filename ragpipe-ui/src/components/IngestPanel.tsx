"use client";

import { useState, useRef } from "react";
import { Upload, FileText, X, Trash2, Plus, Loader2 } from "lucide-react";
import {
  uploadFiles,
  ingestDocuments,
  clearIndex,
  StatsResponse,
} from "@/lib/api";

interface IngestPanelProps {
  open: boolean;
  onClose: () => void;
  stats: StatsResponse;
  onIngested: () => void;
}

export function IngestPanel({ open, onClose, stats, onIngested }: IngestPanelProps) {
  const [tab, setTab] = useState<"upload" | "paste">("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [pasteText, setPasteText] = useState("");
  const [pasteName, setPasteName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!open) return null;

  const handleFiles = (incoming: FileList | File[]) => {
    setFiles((prev) => [...prev, ...Array.from(incoming)]);
    setError(null);
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError("Add at least one file first");
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const res = await uploadFiles(files);
      setSuccess(`Ingested ${res.documents} documents → ${res.chunks} chunks`);
      setFiles([]);
      onIngested();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const handlePasteIngest = async () => {
    if (!pasteText.trim()) {
      setError("Paste some text first");
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const res = await ingestDocuments([
        {
          content: pasteText,
          metadata: { source: pasteName || "pasted-text", filename: pasteName || "pasted.txt" },
        },
      ]);
      setSuccess(`Ingested ${res.documents} document → ${res.chunks} chunks`);
      setPasteText("");
      setPasteName("");
      onIngested();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ingest failed");
    } finally {
      setBusy(false);
    }
  };

  const handleClear = async () => {
    if (!confirm(`Clear all ${stats.documents} documents and ${stats.chunks} chunks? This cannot be undone.`)) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await clearIndex();
      setSuccess("Index cleared");
      onIngested();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Clear failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in modal-backdrop"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="modal-panel relative w-full max-w-2xl max-h-[85vh] rounded-2xl flex flex-col overflow-hidden animate-slide-up"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div>
            <h2 className="text-base font-semibold">Knowledge Base</h2>
            <p className="text-xs text-foreground-muted mt-0.5">
              {stats.documents} documents · {stats.chunks} chunks indexed
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-surface-3 text-foreground-muted hover:text-foreground"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border px-5">
          <TabButton active={tab === "upload"} onClick={() => setTab("upload")}>
            <Upload className="w-3.5 h-3.5" /> Upload files
          </TabButton>
          <TabButton active={tab === "paste"} onClick={() => setTab("paste")}>
            <FileText className="w-3.5 h-3.5" /> Paste text
          </TabButton>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {tab === "upload" && (
            <div className="space-y-3">
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
                }}
                onClick={() => fileInputRef.current?.click()}
                className={`flex flex-col items-center justify-center py-10 px-6 rounded-xl border-2 border-dashed cursor-pointer transition-colors ${
                  dragOver
                    ? "border-accent bg-accent/5"
                    : "border-border hover:border-border-strong bg-surface-2/50"
                }`}
              >
                <Upload className="w-8 h-8 text-foreground-muted mb-3" />
                <p className="text-sm font-medium">Drop files here or click to browse</p>
                <p className="text-xs text-foreground-subtle mt-1.5">
                  PDF, DOCX, TXT, MD, HTML, CSV
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.md,.html,.htm,.csv,.json,.rst,.log"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files) handleFiles(e.target.files);
                    e.target.value = "";
                  }}
                />
              </div>

              {files.length > 0 && (
                <div className="space-y-1.5">
                  {files.map((f, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-2 text-sm"
                    >
                      <FileText className="w-3.5 h-3.5 text-foreground-muted shrink-0" />
                      <span className="flex-1 truncate">{f.name}</span>
                      <span className="text-xs text-foreground-subtle tabular-nums">
                        {fmtBytes(f.size)}
                      </span>
                      <button
                        onClick={() => removeFile(i)}
                        className="p-1 rounded hover:bg-surface-3 text-foreground-muted hover:text-destructive"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {tab === "paste" && (
            <div className="space-y-3">
              <input
                type="text"
                value={pasteName}
                onChange={(e) => setPasteName(e.target.value)}
                placeholder="Document name (optional)"
                className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm focus:outline-none focus:border-accent"
              />
              <textarea
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
                placeholder="Paste your text content here..."
                rows={12}
                className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm focus:outline-none focus:border-accent resize-none font-mono"
              />
              <div className="text-xs text-foreground-subtle text-right tabular-nums">
                {pasteText.length} chars · ~{Math.ceil(pasteText.length / 4)} tokens
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 px-5 py-3 border-t border-border bg-surface-2/30">
          {error && <p className="text-xs text-destructive flex-1">{error}</p>}
          {success && <p className="text-xs text-success flex-1">{success}</p>}
          {!error && !success && <div className="flex-1" />}

          {stats.documents > 0 && (
            <button
              onClick={handleClear}
              disabled={busy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-foreground-muted hover:text-destructive hover:bg-destructive/10 disabled:opacity-50 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" /> Clear all
            </button>
          )}

          <button
            onClick={tab === "upload" ? handleUpload : handlePasteIngest}
            disabled={busy || (tab === "upload" ? files.length === 0 : !pasteText.trim())}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover text-accent-foreground text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
          >
            {busy ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Ingesting...
              </>
            ) : (
              <>
                <Plus className="w-4 h-4" /> Ingest
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function TabButton({
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
      className={`flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium border-b-2 transition-colors -mb-px ${
        active
          ? "border-accent text-foreground"
          : "border-transparent text-foreground-muted hover:text-foreground"
      }`}
    >
      {children}
    </button>
  );
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
