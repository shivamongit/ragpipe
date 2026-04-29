"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Check, Cpu, Cloud, Zap, AlertCircle } from "lucide-react";
import { ModelInfo, listModels } from "@/lib/api";

interface ModelPickerProps {
  value: { provider: string; model: string } | null;
  onChange: (selection: { provider: string; model: string }) => void;
}

const PROVIDER_COLORS: Record<string, string> = {
  openai: "var(--c-openai)",
  anthropic: "var(--c-anthropic)",
  google: "var(--c-google)",
  groq: "var(--c-groq)",
  cohere: "var(--c-cohere)",
  mistral: "var(--c-mistral)",
  ollama: "var(--c-ollama)",
};

export function ModelPicker({ value, onChange }: ModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [filter, setFilter] = useState("");
  const [showUnavailable, setShowUnavailable] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  useEffect(() => {
    const onClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  const visibleModels = models.filter((m) => {
    if (!showUnavailable && !m.available) return false;
    if (!filter) return true;
    const f = filter.toLowerCase();
    return (
      m.name.toLowerCase().includes(f) ||
      m.id.toLowerCase().includes(f) ||
      m.provider_name.toLowerCase().includes(f) ||
      m.tags.some((t) => t.toLowerCase().includes(f))
    );
  });

  const grouped: Record<string, ModelInfo[]> = {};
  for (const m of visibleModels) {
    if (!grouped[m.provider_name]) grouped[m.provider_name] = [];
    grouped[m.provider_name].push(m);
  }

  const current = value ? models.find((m) => m.id === value.model && m.provider === value.provider) : null;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-2 hover:bg-surface-3 border border-border text-sm text-foreground-muted hover:text-foreground transition-colors"
      >
        {current ? (
          <>
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ background: PROVIDER_COLORS[current.provider] || "var(--accent)" }}
            />
            <span className="font-medium text-foreground">{current.name}</span>
            <span className="text-foreground-subtle text-xs hidden sm:inline">
              {current.provider_name}
            </span>
          </>
        ) : (
          <>
            <Cpu className="w-3.5 h-3.5" />
            <span>Select model</span>
          </>
        )}
        <ChevronDown className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-[420px] max-h-[520px] glass rounded-xl shadow-2xl shadow-black/40 z-50 overflow-hidden flex flex-col animate-fade-in">
          <div className="px-3 py-2.5 border-b border-border">
            <input
              autoFocus
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Search models, providers, tags..."
              className="w-full bg-surface-3 border border-border rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-accent"
            />
            <label className="flex items-center gap-2 mt-2 text-xs text-foreground-muted cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showUnavailable}
                onChange={(e) => setShowUnavailable(e.target.checked)}
                className="accent-indigo-500"
              />
              Show unavailable (no API key / not installed)
            </label>
          </div>

          <div className="overflow-y-auto flex-1">
            {Object.keys(grouped).length === 0 ? (
              <div className="px-4 py-8 text-center text-xs text-foreground-subtle">
                No models match "{filter}"
              </div>
            ) : (
              Object.entries(grouped).map(([providerName, list]) => (
                <div key={providerName} className="py-1">
                  <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] uppercase tracking-wider text-foreground-subtle font-medium">
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ background: PROVIDER_COLORS[list[0].provider] || "var(--accent)" }}
                    />
                    {providerName}
                  </div>
                  {list.map((m) => {
                    const selected =
                      value?.provider === m.provider && value.model === m.id;
                    return (
                      <button
                        key={`${m.provider}-${m.id}`}
                        onClick={() => {
                          onChange({ provider: m.provider, model: m.id });
                          setOpen(false);
                          setFilter("");
                        }}
                        disabled={!m.available}
                        className={`w-full text-left px-3 py-2 flex items-start gap-2.5 transition-colors ${
                          selected
                            ? "bg-accent/10 text-accent-foreground"
                            : m.available
                            ? "hover:bg-surface-3"
                            : "opacity-50 cursor-not-allowed"
                        }`}
                      >
                        <div
                          className="w-7 h-7 rounded-md shrink-0 flex items-center justify-center text-[10px] font-bold mt-0.5"
                          style={{
                            background: `${PROVIDER_COLORS[m.provider]}22`,
                            color: PROVIDER_COLORS[m.provider],
                          }}
                        >
                          {providerInitials(m.provider)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-foreground truncate">
                              {m.name}
                            </span>
                            {selected && <Check className="w-3.5 h-3.5 text-accent shrink-0" />}
                            {!m.available && (
                              <AlertCircle
                                className="w-3.5 h-3.5 text-warning shrink-0"
                                aria-label="Unavailable"
                              />
                            )}
                          </div>
                          <div className="flex items-center gap-2 mt-0.5 text-[11px] text-foreground-subtle">
                            <span className="truncate">{m.description}</span>
                          </div>
                          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                            {m.tags.slice(0, 4).map((t) => (
                              <span
                                key={t}
                                className="px-1.5 py-0 rounded text-[9px] bg-surface-3 text-foreground-muted uppercase tracking-wider"
                              >
                                {t}
                              </span>
                            ))}
                            <span className="ml-auto text-[10px] tabular-nums text-foreground-subtle">
                              {m.context_window >= 1_000_000
                                ? `${m.context_window / 1_000_000}M`
                                : `${(m.context_window / 1000) | 0}K`}{" "}
                              ctx
                            </span>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              ))
            )}
          </div>

          <div className="px-3 py-2 border-t border-border text-[10px] text-foreground-subtle flex items-center justify-between">
            <span>{visibleModels.length} models</span>
            <span className="flex items-center gap-1">
              <Zap className="w-3 h-3" /> Switch any time
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function providerInitials(p: string): string {
  switch (p) {
    case "openai": return "AI";
    case "anthropic": return "AN";
    case "google": return "G";
    case "groq": return "Q";
    case "cohere": return "CO";
    case "mistral": return "M";
    case "ollama": return "OL";
    default: return p.slice(0, 2).toUpperCase();
  }
}
