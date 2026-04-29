"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  X,
  Network,
  RefreshCw,
  Loader2,
  AlertCircle,
  Search,
  Sparkles,
} from "lucide-react";
import {
  getKnowledgeGraph,
  KnowledgeGraphResponse,
  GraphEntity,
  GraphTriple,
} from "@/lib/api";

interface GraphPanelProps {
  open: boolean;
  onClose: () => void;
}

interface Node {
  id: string;
  name: string;
  mentions: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

interface Edge {
  source: string;
  target: string;
  predicate: string;
}

export function GraphPanel({ open, onClose }: GraphPanelProps) {
  const [graph, setGraph] = useState<KnowledgeGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number | null>(null);
  const nodesRef = useRef<Node[]>([]);
  const edgesRef = useRef<Edge[]>([]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getKnowledgeGraph({ maxEntities: 60, maxTriples: 200 });
      setGraph(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load graph");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) refresh();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [open, refresh]);

  // Build nodes / edges + run force-directed layout
  useEffect(() => {
    if (!graph || !canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
    };
    resize();
    window.addEventListener("resize", resize);

    const w = () => canvas.width / dpr;
    const h = () => canvas.height / dpr;
    const cx = () => w() / 2;
    const cy = () => h() / 2;

    // Initialize nodes
    const maxMentions = Math.max(...graph.entities.map((e) => e.mentions), 1);
    nodesRef.current = graph.entities.map((e: GraphEntity, i: number) => {
      const angle = (i / Math.max(graph.entities.length, 1)) * Math.PI * 2;
      const r = Math.min(w(), h()) * 0.35;
      return {
        id: e.id,
        name: e.name,
        mentions: e.mentions,
        x: cx() + Math.cos(angle) * r,
        y: cy() + Math.sin(angle) * r,
        vx: 0,
        vy: 0,
        radius: 4 + (e.mentions / maxMentions) * 14,
      };
    });

    edgesRef.current = graph.triples
      .map((t: GraphTriple): Edge => ({
        source: t.subject.toLowerCase().trim(),
        target: t.object.toLowerCase().trim(),
        predicate: t.predicate,
      }))
      .filter(
        (e) =>
          nodesRef.current.find((n) => n.id === e.source) &&
          nodesRef.current.find((n) => n.id === e.target),
      );

    const ctx = canvas.getContext("2d")!;

    const tick = () => {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const W = w();
      const H = h();
      const CX = cx();
      const CY = cy();

      // Forces
      // 1. Repulsion between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d2 = dx * dx + dy * dy + 0.01;
          const d = Math.sqrt(d2);
          const force = 1800 / d2;
          const fx = (dx / d) * force;
          const fy = (dy / d) * force;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }
      }

      // 2. Spring on edges
      const idMap = new Map(nodes.map((n) => [n.id, n]));
      for (const e of edges) {
        const a = idMap.get(e.source);
        const b = idMap.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const target = 110;
        const k = 0.012;
        const fx = (dx / d) * (d - target) * k;
        const fy = (dy / d) * (d - target) * k;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }

      // 3. Gravity to center
      for (const n of nodes) {
        n.vx += (CX - n.x) * 0.002;
        n.vy += (CY - n.y) * 0.002;
        // damping
        n.vx *= 0.82;
        n.vy *= 0.82;
        n.x += n.vx;
        n.y += n.vy;
        // bounds
        n.x = Math.max(n.radius + 4, Math.min(W - n.radius - 4, n.x));
        n.y = Math.max(n.radius + 4, Math.min(H - n.radius - 4, n.y));
      }

      // ─ Render ─
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // Subtle grid background
      ctx.strokeStyle = "rgba(255,255,255,0.025)";
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < W; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }
      for (let y = 0; y < H; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      // Edges with gradient
      const searchLower = search.toLowerCase().trim();
      for (const e of edges) {
        const a = idMap.get(e.source);
        const b = idMap.get(e.target);
        if (!a || !b) continue;
        const dim =
          (searchLower &&
            !a.name.toLowerCase().includes(searchLower) &&
            !b.name.toLowerCase().includes(searchLower)) ||
          (hoverNode && hoverNode !== a.id && hoverNode !== b.id);
        const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        grad.addColorStop(0, dim ? "rgba(99,102,241,0.06)" : "rgba(99,102,241,0.55)");
        grad.addColorStop(1, dim ? "rgba(168,85,247,0.06)" : "rgba(168,85,247,0.55)");
        ctx.strokeStyle = grad;
        ctx.lineWidth = dim ? 0.6 : 1.2;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      // Nodes
      for (const n of nodes) {
        const matched = !searchLower || n.name.toLowerCase().includes(searchLower);
        const isHover = hoverNode === n.id;
        const dim = !matched || (hoverNode && !isHover);

        // Glow
        if (isHover || (searchLower && matched)) {
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.radius * 4);
          g.addColorStop(0, "rgba(168,85,247,0.55)");
          g.addColorStop(1, "rgba(168,85,247,0)");
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius * 4, 0, Math.PI * 2);
          ctx.fill();
        }

        // Main circle gradient
        const circ = ctx.createRadialGradient(
          n.x - n.radius * 0.3,
          n.y - n.radius * 0.3,
          0,
          n.x,
          n.y,
          n.radius,
        );
        if (dim) {
          circ.addColorStop(0, "rgba(99,102,241,0.25)");
          circ.addColorStop(1, "rgba(168,85,247,0.10)");
        } else {
          circ.addColorStop(0, "#a5b4fc");
          circ.addColorStop(1, "#7c3aed");
        }
        ctx.fillStyle = circ;
        ctx.strokeStyle = dim ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.20)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Label (only for larger / hovered / matched)
        if (n.radius > 9 || isHover || (searchLower && matched)) {
          ctx.fillStyle = dim ? "rgba(245,246,250,0.30)" : "rgba(245,246,250,0.95)";
          ctx.font = "11px system-ui, -apple-system, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(n.name.slice(0, 22), n.x, n.y + n.radius + 4);
        }
      }

      animRef.current = requestAnimationFrame(tick);
    };

    tick();

    // Hover detection
    const handleMove = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      let found: string | null = null;
      for (const n of nodesRef.current) {
        const dx = mx - n.x;
        const dy = my - n.y;
        if (dx * dx + dy * dy <= (n.radius + 4) ** 2) {
          found = n.id;
          break;
        }
      }
      setHoverNode(found);
      canvas.style.cursor = found ? "pointer" : "default";
    };
    canvas.addEventListener("mousemove", handleMove);

    return () => {
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("mousemove", handleMove);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [graph, search, hoverNode]);

  if (!open) return null;

  const filteredEntities = graph
    ? graph.entities.filter((e) =>
        e.name.toLowerCase().includes(search.toLowerCase()),
      )
    : [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in modal-backdrop"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="modal-panel relative w-full max-w-6xl h-[88vh] rounded-2xl flex overflow-hidden animate-slide-up"
      >
        {/* Side: stats + entity list */}
        <div className="w-72 shrink-0 border-r border-border bg-surface-2/40 flex flex-col">
          <div className="px-4 py-4 border-b border-border">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-grad-brand flex items-center justify-center shadow-md">
                <Network className="w-3.5 h-3.5 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-sm font-semibold leading-none">Knowledge Graph</h2>
                <p className="text-[10px] text-foreground-subtle mt-1 uppercase tracking-wider">
                  Auto-extracted entities
                </p>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-1 p-3">
            <Stat label="Entities" value={graph?.entity_count ?? 0} />
            <Stat label="Triples" value={graph?.triple_count ?? 0} />
            <Stat label="Docs" value={graph?.documents_indexed ?? 0} />
          </div>

          {/* Search */}
          <div className="px-3 pb-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-foreground-subtle" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter entities..."
                className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-surface-3 border border-border text-xs focus:outline-none focus:border-accent"
              />
            </div>
          </div>

          {/* List */}
          <div className="flex-1 overflow-y-auto px-2 pb-3">
            {loading && !graph ? (
              <div className="flex items-center justify-center py-10">
                <Loader2 className="w-5 h-5 animate-spin text-foreground-muted" />
              </div>
            ) : filteredEntities.length === 0 ? (
              <div className="px-3 py-8 text-center text-xs text-foreground-subtle">
                {graph?.entity_count === 0
                  ? "No entities found. Ingest documents with proper nouns."
                  : "No matches"}
              </div>
            ) : (
              filteredEntities.map((e) => (
                <button
                  key={e.id}
                  onMouseEnter={() => setHoverNode(e.id)}
                  onMouseLeave={() => setHoverNode(null)}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 mb-0.5 rounded-md text-left text-xs transition-colors ${
                    hoverNode === e.id
                      ? "bg-accent/15 text-foreground"
                      : "text-foreground-muted hover:bg-surface-3"
                  }`}
                >
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{
                      background:
                        "linear-gradient(135deg, #818cf8, #c084fc)",
                    }}
                  />
                  <span className="flex-1 truncate">{e.name}</span>
                  <span className="text-[10px] text-foreground-subtle tabular-nums">
                    {e.mentions}×
                  </span>
                </button>
              ))
            )}
          </div>

          <div className="border-t border-border px-3 py-2.5 flex items-center gap-2">
            <button
              onClick={refresh}
              disabled={loading}
              className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-xs text-foreground-muted hover:text-foreground hover:bg-surface-3 disabled:opacity-50 transition-colors"
            >
              {loading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <RefreshCw className="w-3.5 h-3.5" />
              )}
              Refresh
            </button>
          </div>
        </div>

        {/* Main: canvas */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center justify-between px-5 py-3 border-b border-border">
            <div className="flex items-center gap-2 text-xs text-foreground-muted">
              <Sparkles className="w-3.5 h-3.5 text-accent-hover" />
              <span>Force-directed layout · hover for details</span>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-surface-3 text-foreground-muted hover:text-foreground transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div ref={containerRef} className="flex-1 relative bg-background-2/60">
            {error && (
              <div className="absolute top-4 left-4 right-4 z-10 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/20 text-xs text-destructive flex items-center gap-2">
                <AlertCircle className="w-3.5 h-3.5" />
                {error}
              </div>
            )}
            {!loading && graph && graph.entity_count === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center max-w-sm px-6">
                  <Network className="w-10 h-10 text-foreground-subtle mx-auto mb-3" />
                  <h3 className="text-sm font-medium mb-1">No graph yet</h3>
                  <p className="text-xs text-foreground-muted">
                    Ingest documents with named entities (people, places,
                    organizations) and the graph will populate automatically.
                  </p>
                </div>
              </div>
            )}
            <canvas ref={canvasRef} className="absolute inset-0" />

            {/* Hover tooltip */}
            {hoverNode && graph && (
              <div className="absolute bottom-4 left-4 px-3 py-2 rounded-lg glass-strong text-xs animate-fade-in pointer-events-none">
                {(() => {
                  const ent = graph.entities.find((e) => e.id === hoverNode);
                  if (!ent) return null;
                  const related = graph.triples
                    .filter(
                      (t) =>
                        t.subject.toLowerCase().trim() === hoverNode ||
                        t.object.toLowerCase().trim() === hoverNode,
                    )
                    .slice(0, 5);
                  return (
                    <>
                      <div className="font-semibold text-foreground">
                        {ent.name}
                      </div>
                      <div className="text-foreground-subtle text-[10px] mb-1.5">
                        {ent.mentions} mentions · {ent.neighbors} neighbors
                      </div>
                      {related.map((r, i) => (
                        <div key={i} className="text-foreground-muted text-[11px]">
                          <span className="text-accent-hover">{r.predicate.replace(/_/g, " ")}</span>{" "}
                          → {r.subject.toLowerCase().trim() === hoverNode ? r.object : r.subject}
                        </div>
                      ))}
                    </>
                  );
                })()}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-surface-3/60 border border-border px-2 py-1.5 text-center">
      <div className="text-sm font-semibold tabular-nums">{value}</div>
      <div className="text-[9px] uppercase tracking-wider text-foreground-subtle mt-0.5">
        {label}
      </div>
    </div>
  );
}
