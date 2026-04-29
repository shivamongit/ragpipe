"use client";

import { useEffect, useState } from "react";
import {
  Plus,
  MessageSquare,
  Trash2,
  Edit3,
  Settings,
  Database,
  ExternalLink,
  Sparkles,
  Check,
  X,
} from "lucide-react";
import {
  Conversation,
  listConversations,
  createConversation,
  deleteConversation as apiDeleteConversation,
  renameConversation,
} from "@/lib/api";

interface SidebarProps {
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onOpenSettings: () => void;
  onOpenIngest: () => void;
  stats: { documents: number; chunks: number };
  health: "online" | "offline" | "unknown";
  refreshKey: number;
}

export function Sidebar({
  activeId,
  onSelect,
  onNewChat,
  onOpenSettings,
  onOpenIngest,
  stats,
  health,
  refreshKey,
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    listConversations()
      .then((c) => {
        if (mounted) setConversations(c);
      })
      .catch(() => {
        if (mounted) setConversations([]);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [refreshKey]);

  const handleNewChat = async () => {
    onNewChat();
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this conversation?")) return;
    try {
      await apiDeleteConversation(id);
      setConversations((c) => c.filter((x) => x.id !== id));
      if (activeId === id) onNewChat();
    } catch (err) {
      console.error(err);
    }
  };

  const handleRename = async (id: string) => {
    const trimmed = editTitle.trim();
    if (!trimmed) {
      setEditingId(null);
      return;
    }
    try {
      await renameConversation(id, trimmed);
      setConversations((c) => c.map((x) => (x.id === id ? { ...x, title: trimmed } : x)));
    } finally {
      setEditingId(null);
    }
  };

  const startEdit = (conv: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const groupedConversations = groupByDate(conversations);

  return (
    <aside className="w-72 shrink-0 h-full flex flex-col border-r border-border bg-surface/40 backdrop-blur-sm relative z-10">
      {/* Brand */}
      <div className="px-4 py-4 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-sm font-semibold leading-none tracking-tight">ragpipe</h1>
            <p className="text-[10px] text-foreground-subtle mt-1 uppercase tracking-wider">
              RAG Studio v3.1
            </p>
          </div>
        </div>
      </div>

      {/* New chat */}
      <div className="p-3">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-accent-foreground text-sm font-medium transition-all shadow-sm hover:shadow-md hover:shadow-accent/20"
        >
          <Plus className="w-4 h-4" />
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <span className="dot-loader">
              <span /> <span /> <span />
            </span>
          </div>
        ) : conversations.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-foreground-subtle">
            No conversations yet. Start a new chat to begin.
          </div>
        ) : (
          Object.entries(groupedConversations).map(([label, convs]) => (
            <div key={label} className="mb-3">
              <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-foreground-subtle font-medium">
                {label}
              </div>
              {convs.map((conv) => (
                <div
                  key={conv.id}
                  onClick={() => onSelect(conv.id)}
                  className={`group flex items-center gap-2 px-2.5 py-2 mb-0.5 rounded-md cursor-pointer text-sm transition-colors ${
                    activeId === conv.id
                      ? "bg-surface-3 text-foreground"
                      : "text-foreground-muted hover:bg-surface-2 hover:text-foreground"
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5 shrink-0 opacity-70" />
                  {editingId === conv.id ? (
                    <input
                      autoFocus
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onBlur={() => handleRename(conv.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleRename(conv.id);
                        if (e.key === "Escape") setEditingId(null);
                      }}
                      className="flex-1 min-w-0 bg-surface-3 border border-border-strong rounded px-1.5 py-0.5 text-sm focus:outline-none focus:border-accent"
                    />
                  ) : (
                    <>
                      <span className="flex-1 min-w-0 truncate">{conv.title}</span>
                      <div className="hidden group-hover:flex items-center gap-0.5 shrink-0">
                        <button
                          onClick={(e) => startEdit(conv, e)}
                          className="p-1 rounded hover:bg-surface-hover text-foreground-subtle hover:text-foreground"
                          title="Rename"
                        >
                          <Edit3 className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => handleDelete(conv.id, e)}
                          className="p-1 rounded hover:bg-destructive/10 text-foreground-subtle hover:text-destructive"
                          title="Delete"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Footer: stats + health + settings */}
      <div className="border-t border-border px-3 py-3 space-y-2">
        <button
          onClick={onOpenIngest}
          className="w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg bg-surface-2 hover:bg-surface-3 text-foreground-muted hover:text-foreground text-xs transition-colors"
        >
          <span className="flex items-center gap-2">
            <Database className="w-3.5 h-3.5" />
            <span>Knowledge Base</span>
          </span>
          <span className="text-[10px] tabular-nums">
            {stats.documents} <span className="text-foreground-subtle">docs</span> · {stats.chunks}{" "}
            <span className="text-foreground-subtle">chunks</span>
          </span>
        </button>

        <div className="flex items-center justify-between gap-2 px-2.5 py-1.5 text-[11px] text-foreground-subtle">
          <span className="flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                health === "online"
                  ? "bg-success pulse-dot"
                  : health === "offline"
                  ? "bg-destructive"
                  : "bg-foreground-subtle"
              }`}
            />
            <span className="capitalize">{health}</span>
          </span>
          <a
            href="https://github.com/shivamongit/ragpipe"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
            title="GitHub"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg hover:bg-surface-2 text-foreground-muted hover:text-foreground text-sm transition-colors"
        >
          <Settings className="w-4 h-4" />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  );
}

function groupByDate(convs: Conversation[]): Record<string, Conversation[]> {
  const now = Date.now();
  const groups: Record<string, Conversation[]> = {};
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const yesterdayStart = todayStart.getTime() - 86_400_000;
  const weekStart = todayStart.getTime() - 6 * 86_400_000;
  const monthStart = todayStart.getTime() - 30 * 86_400_000;

  for (const c of convs) {
    let label: string;
    if (c.updated_at >= todayStart.getTime()) label = "Today";
    else if (c.updated_at >= yesterdayStart) label = "Yesterday";
    else if (c.updated_at >= weekStart) label = "Previous 7 days";
    else if (c.updated_at >= monthStart) label = "Previous 30 days";
    else label = "Older";

    if (!groups[label]) groups[label] = [];
    groups[label].push(c);
  }
  return groups;
}
