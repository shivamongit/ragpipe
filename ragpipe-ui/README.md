# ragpipe UI

Modern, Ollama-style dark chat interface for the ragpipe Context Engineering Platform.

Built with **Next.js 16**, **Tailwind CSS v4**, and **Lucide icons**.

![ragpipe UI](https://img.shields.io/badge/theme-dark-000000) ![Next.js](https://img.shields.io/badge/Next.js-16-black) ![Tailwind](https://img.shields.io/badge/Tailwind-v4-38bdf8)

## Features

- **Dark theme chat** — Clean, minimal Ollama-style interface
- **Streaming responses** — Real-time token streaming via WebSocket
- **Source citations** — Expandable source cards with relevance scores
- **Document ingestion** — Drag & drop files or paste text
- **Pipeline stats** — Live document/chunk counts from the backend
- **Server health** — Auto-polling connection status indicator
- **Settings panel** — Configure API URL, API key, and top-K
- **Quick suggestions** — Clickable starter prompts
- **Response metadata** — Model name, token count, latency per response

## Getting Started

### 1. Start the ragpipe backend

```bash
cd /path/to/ragpipe
pip install -e ".[server]"
python -m ragpipe serve --config pipeline.yml --port 8000
```

### 2. Start the UI

```bash
cd ragpipe-ui
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Configure

Click the **⚙ Settings** button to set:
- **API URL** — defaults to `http://localhost:8000`
- **API Key** — leave empty if not using auth
- **Top K** — number of retrieved sources per query

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | ragpipe backend URL |

## API Endpoints Used

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Send a question, get answer + sources |
| `POST` | `/ingest` | Ingest documents |
| `GET` | `/stats` | Document/chunk counts |
| `GET` | `/health` | Server health check |
| `WS` | `/query/stream` | Streaming token responses |

## Tech Stack

- **Next.js 16** — React framework with App Router
- **Tailwind CSS v4** — Utility-first styling
- **Lucide React** — Beautiful, consistent icons
- **Geist Font** — Clean sans-serif + monospace
