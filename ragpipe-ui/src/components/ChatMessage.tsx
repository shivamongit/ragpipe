"use client";

import { useState } from "react";
import {
  User,
  Sparkles,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Clock,
  Hash,
} from "lucide-react";
import { Source } from "@/lib/api";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  model?: string;
  tokensUsed?: number;
  latencyMs?: number;
  isStreaming?: boolean;
  timestamp: Date;
}

export function ChatMessage({ message }: { message: UIMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`group animate-fade-in ${isUser ? "" : ""}`}>
      <div className="max-w-[850px] mx-auto px-6 py-5 flex gap-4">
        {/* Avatar */}
        <div
          className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center mt-0.5 ${
            isUser
              ? "bg-surface-3 text-foreground-muted"
              : "bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-md shadow-indigo-500/20"
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
        </div>

        {/* Body */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs font-semibold tracking-wide">
              {isUser ? "You" : "Assistant"}
            </span>
            {message.model && !isUser && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-3 text-foreground-muted font-mono">
                {message.model}
              </span>
            )}
            {!isUser && message.latencyMs !== undefined && (
              <span className="text-[10px] text-foreground-subtle flex items-center gap-1 tabular-nums">
                <Clock className="w-3 h-3" />
                {message.latencyMs.toFixed(0)}ms
              </span>
            )}
            {!isUser && message.tokensUsed !== undefined && message.tokensUsed > 0 && (
              <span className="text-[10px] text-foreground-subtle flex items-center gap-1 tabular-nums">
                <Hash className="w-3 h-3" />
                {message.tokensUsed} tok
              </span>
            )}
          </div>

          <MessageContent content={message.content} streaming={message.isStreaming} />

          {!isUser && message.sources && message.sources.length > 0 && (
            <SourcesList sources={message.sources} />
          )}

          {!isUser && !message.isStreaming && message.content && (
            <CopyButton text={message.content} />
          )}
        </div>
      </div>
    </div>
  );
}

function MessageContent({ content, streaming }: { content: string; streaming?: boolean }) {
  // Light-weight markdown rendering: bold, italic, code, lists, headings, blockquotes
  const html = renderLightMarkdown(content);
  return (
    <div
      className={`prose-chat text-[15px] text-foreground ${streaming ? "typing-cursor" : ""}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function renderLightMarkdown(text: string): string {
  if (!text) return "";

  // Escape HTML
  let s = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks ```lang ... ```
  s = s.replace(/```(\w+)?\n([\s\S]*?)```/g, (_m, _lang, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  // Inline code
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Headings
  s = s.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  s = s.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Bold + italic
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");

  // Blockquotes
  s = s.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  // Source citations [Source N] -> styled span
  s = s.replace(/\[Source (\d+)\]/g, '<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[11px] bg-accent/15 text-accent-hover font-medium font-mono">$1</span>');

  // Lists
  // Unordered
  s = s.replace(/((?:^- .+\n?)+)/gm, (block) => {
    const items = block.trim().split("\n").map((l) => l.replace(/^- /, "")).map((i) => `<li>${i}</li>`).join("");
    return `<ul>${items}</ul>`;
  });
  // Ordered
  s = s.replace(/((?:^\d+\. .+\n?)+)/gm, (block) => {
    const items = block.trim().split("\n").map((l) => l.replace(/^\d+\. /, "")).map((i) => `<li>${i}</li>`).join("");
    return `<ol>${items}</ol>`;
  });

  // Paragraphs (split on blank line)
  const blocks = s.split(/\n\n+/).map((b) => {
    const trimmed = b.trim();
    if (!trimmed) return "";
    // skip if already a block element
    if (/^<(h[1-6]|ul|ol|pre|blockquote)/.test(trimmed)) return trimmed;
    // line breaks within paragraphs
    return `<p>${trimmed.replace(/\n/g, "<br/>")}</p>`;
  });
  return blocks.join("\n");
}

function SourcesList({ sources }: { sources: Source[] }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded((e) => !e)}
        className="flex items-center gap-1.5 text-xs text-foreground-muted hover:text-foreground transition-colors"
      >
        {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        <span>{sources.length} source{sources.length !== 1 ? "s" : ""}</span>
      </button>
      {expanded && (
        <div className="mt-2 space-y-1.5 animate-fade-in">
          {sources.map((s, i) => (
            <SourceCard key={i} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceCard({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-border bg-surface-2/60 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-surface-3/50 transition-colors"
      >
        <span className="w-6 h-6 rounded-md bg-accent/15 text-accent-hover text-[11px] font-mono font-semibold flex items-center justify-center shrink-0">
          {source.rank}
        </span>
        <span className="flex-1 min-w-0 text-xs text-foreground-muted truncate">
          {source.text.slice(0, 120)}
          {source.text.length > 120 ? "…" : ""}
        </span>
        <span className="text-[10px] text-foreground-subtle tabular-nums shrink-0">
          {(source.score * 100).toFixed(1)}%
        </span>
        {open ? (
          <ChevronDown className="w-3.5 h-3.5 text-foreground-subtle shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-foreground-subtle shrink-0" />
        )}
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 text-xs text-foreground-muted leading-relaxed border-t border-border bg-surface-2/30 animate-fade-in">
          <pre className="whitespace-pre-wrap font-sans">{source.text}</pre>
          <div className="mt-2 text-[10px] text-foreground-subtle font-mono">
            doc_id: {source.doc_id}
          </div>
        </div>
      )}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="mt-2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 text-[11px] text-foreground-muted hover:text-foreground"
    >
      {copied ? (
        <>
          <Check className="w-3 h-3" /> Copied
        </>
      ) : (
        <>
          <Copy className="w-3 h-3" /> Copy
        </>
      )}
    </button>
  );
}
