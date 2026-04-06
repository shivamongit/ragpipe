"""FastAPI REST API server for ragpipe pipelines."""

from __future__ import annotations

import os
from typing import Any, Optional

from ragpipe.core import Document, Pipeline


def create_app(pipeline: Pipeline | None = None, api_key: str | None = None):
    """Create a FastAPI app wrapping a ragpipe Pipeline.

    Args:
        pipeline: Pre-configured Pipeline instance. If None, must be set later via app.state.pipeline.
        api_key: Optional API key for authentication via X-API-Key header.
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install server dependencies: pip install 'ragpipe[server]'")

    app = FastAPI(
        title="ragpipe",
        description="Production-grade RAG pipeline API",
        version="3.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.pipeline = pipeline
    _api_key = api_key or os.environ.get("RAGPIPE_API_KEY")

    # --- Auth dependency ---
    async def verify_api_key(x_api_key: Optional[str] = Header(None)):
        if _api_key and x_api_key != _api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def get_pipeline() -> Pipeline:
        pipe = app.state.pipeline
        if pipe is None:
            raise HTTPException(status_code=503, detail="Pipeline not configured")
        return pipe

    # --- Request/Response models ---
    class IngestRequest(BaseModel):
        documents: list[dict[str, Any]]

    class IngestResponse(BaseModel):
        documents: int
        chunks: int

    class QueryRequest(BaseModel):
        question: str
        top_k: Optional[int] = None

    class SourceResponse(BaseModel):
        text: str
        doc_id: str
        score: float
        rank: int

    class QueryResponse(BaseModel):
        answer: str
        sources: list[SourceResponse]
        model: str
        tokens_used: int
        latency_ms: float

    class StatsResponse(BaseModel):
        documents: int
        chunks: int

    class EvaluateRequest(BaseModel):
        question: str
        relevant_doc_ids: list[str]
        top_k: Optional[int] = None

    class EvaluateResponse(BaseModel):
        hit_rate: float
        mrr: float
        precision_at_k: float
        recall_at_k: float
        ndcg_at_k: float

    # --- Endpoints ---
    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
    async def ingest(req: IngestRequest):
        pipe = get_pipeline()
        docs = [
            Document(
                content=d.get("content", ""),
                metadata=d.get("metadata", {}),
                doc_id=d.get("doc_id", ""),
            )
            for d in req.documents
        ]
        stats = await pipe.aingest(docs)
        return IngestResponse(**stats)

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
    async def query(req: QueryRequest):
        pipe = get_pipeline()
        result = await pipe.aquery(req.question, top_k=req.top_k)
        sources = [
            SourceResponse(
                text=s.chunk.text[:500],
                doc_id=s.chunk.doc_id,
                score=round(s.score, 4),
                rank=s.rank,
            )
            for s in result.sources
        ]
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            model=result.model,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
        )

    @app.websocket("/query/stream")
    async def query_stream(ws: WebSocket):
        await ws.accept()

        if _api_key:
            try:
                auth = await ws.receive_json()
                if auth.get("api_key") != _api_key:
                    await ws.close(code=4001, reason="Invalid API key")
                    return
            except Exception:
                await ws.close(code=4001, reason="Auth required")
                return

        try:
            while True:
                data = await ws.receive_json()
                question = data.get("question", "")
                top_k = data.get("top_k")

                pipe = get_pipeline()
                async for token in pipe.stream_query(question, top_k=top_k):
                    await ws.send_json({"type": "token", "content": token})

                await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass

    @app.get("/stats", response_model=StatsResponse, dependencies=[Depends(verify_api_key)])
    async def stats():
        pipe = get_pipeline()
        return StatsResponse(documents=pipe.document_count, chunks=pipe.chunk_count)

    @app.delete("/index", dependencies=[Depends(verify_api_key)])
    async def delete_index():
        pipe = get_pipeline()
        pipe._documents.clear()
        return {"status": "ok", "message": "Documents cleared from memory"}

    @app.post("/evaluate", response_model=EvaluateResponse, dependencies=[Depends(verify_api_key)])
    async def evaluate(req: EvaluateRequest):
        from ragpipe.evaluation import hit_rate, mrr, precision_at_k, recall_at_k, ndcg_at_k

        pipe = get_pipeline()
        results = await pipe.aretrieve(req.question, top_k=req.top_k)
        relevant = set(req.relevant_doc_ids)
        k = req.top_k or pipe.top_k

        return EvaluateResponse(
            hit_rate=hit_rate(results, relevant),
            mrr=mrr(results, relevant),
            precision_at_k=precision_at_k(results, relevant, k=k),
            recall_at_k=recall_at_k(results, relevant, k=k),
            ndcg_at_k=ndcg_at_k(results, relevant, k=k),
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
