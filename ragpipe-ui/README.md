# ragpipe Studio

Modern, polished chat UI for the ragpipe RAG framework. Built with **Next.js 16**, **Tailwind CSS v4**, and **Lucide icons**.

![Next.js](https://img.shields.io/badge/Next.js-16-black) ![Tailwind](https://img.shields.io/badge/Tailwind-v4-38bdf8) ![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue)

## Features

- 💬 **Streaming chat** with token-by-token rendering and stop button
- 🔄 **Multi-LLM model picker** — switch between OpenAI, Anthropic, Gemini, Groq, Cohere, Mistral, Ollama at any time
- 💾 **Conversation history** — SQLite-backed sidebar grouped by date with rename + delete
- 📎 **Drag-and-drop ingestion** — PDF, DOCX, TXT, MD, HTML, CSV
- 🔍 **Source citations** — Expandable cards with relevance scores
- 🔑 **API key management** — Per-provider keys stored in browser `localStorage`
- ⚙️ **Settings drawer** — Server URL, retrieval top-K, theme
- 🌒 **Polished dark UI** — Glass effects, gradients, smooth animations
- ⌨️ **Keyboard shortcuts** — Enter to send, Shift+Enter for newline

## Getting Started

The easiest way is the one-command launcher from the repo root:

```bash
cd ..
./start.sh
```

That will install deps, auto-detect a provider, and start both backend and UI together.

### Manual setup

```bash
# 1. Start the backend
cd ..
python launch.py

# 2. Start the UI (in another terminal)
cd ragpipe-ui
npm install
npm run dev
```

Open http://localhost:3000.

## Architecture

```
src/
├── app/
│   ├── globals.css     — design tokens, animations, prose styles
│   ├── layout.tsx      — root layout, metadata, fonts
│   └── page.tsx        — main chat orchestrator
├── components/
│   ├── Sidebar.tsx       — conversation history + brand + footer
│   ├── ModelPicker.tsx   — searchable provider/model dropdown
│   ├── ChatMessage.tsx   — message bubble + sources + markdown
│   ├── IngestPanel.tsx   — file upload + paste-text modal
│   └── SettingsModal.tsx — tabbed settings (providers/server/retrieval)
└── lib/
    └── api.ts          — typed REST + WebSocket client
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | ragpipe backend URL |

## API Endpoints Used

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server liveness |
| `GET` | `/providers` | List LLM providers + availability |
| `GET` | `/models` | List all available models |
| `GET` | `/stats` | Document/chunk counts |
| `POST` | `/upload` | Upload files (multipart) |
| `POST` | `/ingest` | Ingest JSON documents |
| `POST` | `/query` | RAG query (provider/model override) |
| `WS` | `/query/stream` | Streaming RAG query |
| `DELETE` | `/index` | Clear knowledge base |
| `GET/POST/PATCH/DELETE` | `/conversations[/{id}]` | Conversation CRUD |

## Tech Stack

- **Next.js 16** — React 19 + App Router + Turbopack dev
- **Tailwind CSS v4** — utility-first, with custom design tokens
- **Lucide React** — consistent icon set
- **Geist Font** — clean sans-serif + monospace
