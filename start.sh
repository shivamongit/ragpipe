#!/usr/bin/env bash
# ragpipe one-command launcher
# Starts the FastAPI backend (port 8000) and Next.js UI (port 3000) together.
#
# Usage:
#   ./start.sh                       # auto-detect best provider
#   ./start.sh --provider openai     # force a provider
#   ./start.sh --model gpt-5-mini    # force a specific model

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Colors
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
DIM="\033[2m"
NC="\033[0m"

step() { echo -e "${GREEN}▸${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

# ── 1. Check Python deps ─────────────────────────────────────────────────────
step "Checking Python environment..."
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found. Install Python 3.10+"
fi

PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  ${DIM}python ${PYV}${NC}"

if ! python3 -c 'import ragpipe' 2>/dev/null; then
    step "Installing ragpipe..."
    python3 -m pip install -e ".[server,sentence-transformers]" --quiet
fi

if ! python3 -c 'import fastapi' 2>/dev/null; then
    step "Installing FastAPI..."
    python3 -m pip install -e ".[server]" --quiet
fi

# ── 2. Check Node / npm ──────────────────────────────────────────────────────
step "Checking Node environment..."
if ! command -v npm >/dev/null 2>&1; then
    fail "npm not found. Install Node 18+ from https://nodejs.org"
fi
NODE_V=$(node -v)
echo -e "  ${DIM}node ${NODE_V}${NC}"

if [ ! -d "ragpipe-ui/node_modules" ]; then
    step "Installing UI dependencies..."
    (cd ragpipe-ui && npm install --silent)
fi

# ── 3. Stop any previously running services ──────────────────────────────────
step "Stopping any previous instances..."
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true

# ── 4. Start backend ─────────────────────────────────────────────────────────
step "Starting backend (port 8000)..."
python3 launch.py "$@" &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null || true; exit" INT TERM EXIT

# Wait until /health responds
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200; then
        echo -e "  ${GREEN}✓${NC} Backend ready"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        fail "Backend did not start within 30s"
    fi
done

# ── 5. Start UI ──────────────────────────────────────────────────────────────
step "Starting UI (port 3000)..."
(cd ragpipe-ui && npm run dev -- -p 3000) &
UI_PID=$!
trap "kill $BACKEND_PID $UI_PID 2>/dev/null || true; exit" INT TERM EXIT

# ── 6. Done ──────────────────────────────────────────────────────────────────
sleep 3
echo ""
echo -e "${GREEN}┌─────────────────────────────────────────────${NC}"
echo -e "${GREEN}│${NC}  ragpipe is running!"
echo -e "${GREEN}│${NC}"
echo -e "${GREEN}│${NC}  ${YELLOW}UI:${NC}      http://localhost:3000"
echo -e "${GREEN}│${NC}  ${YELLOW}API:${NC}     http://localhost:8000"
echo -e "${GREEN}│${NC}  ${YELLOW}Docs:${NC}    http://localhost:8000/docs"
echo -e "${GREEN}│${NC}"
echo -e "${GREEN}│${NC}  Press ${RED}Ctrl-C${NC} to stop both services"
echo -e "${GREEN}└─────────────────────────────────────────────${NC}"
echo ""

wait
