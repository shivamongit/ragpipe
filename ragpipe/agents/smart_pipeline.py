"""SmartPipeline — composable intelligence orchestrator for ragpipe.

Wires every ragpipe module into a single ``.query()`` call that handles:

1. **Guardrails** — PII redaction, prompt-injection detection, topic filtering
2. **Cache** — semantic-similarity cache lookup
3. **Memory** — multi-turn conversation context
4. **Routing** — direct / single / multi-step / summarise strategy selection
5. **Retrieval** — adaptive strategy execution via the core Pipeline
6. **Verification** — claim-level hallucination detection
7. **Cache storage** — persist the answer for future queries

Usage:
    from ragpipe.agents import SmartPipeline

    smart = SmartPipeline(
        pipeline=my_pipeline,
        guardrails=[PIIRedactor(), PromptInjectionDetector()],
        cache=SemanticCache(embed_fn=embed, threshold=0.95),
        memory=ConversationMemory(contextualize_fn=llm),
        verifier=AnswerVerifier(verify_fn=llm),
    )
    result = smart.query("What is FAISS?")
    print(result.answer, result.confidence, result.route_taken)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SmartResult:
    """Result from SmartPipeline execution."""
    answer: str
    confidence: float = 0.0
    sources: list = field(default_factory=list)
    guardrail_checks: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    route_taken: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SmartPipeline:
    """Composable intelligence orchestrator that wires all ragpipe modules together.

    Each module is optional — ``SmartPipeline`` gracefully degrades when components
    are absent so you can adopt them incrementally.  All heavy work is delegated to
    the injected objects; this class is pure orchestration logic.
    """

    def __init__(
        self,
        pipeline: Any = None,
        guardrails: Optional[list] = None,
        cache: Any = None,
        memory: Any = None,
        router: Any = None,
        verifier: Any = None,
        tracer: Any = None,
        pii_redactor: Any = None,
        on_guardrail_fail: str = "block",
        min_confidence: float = 0.0,
    ):
        self.pipeline = pipeline
        self.guardrails = guardrails or []
        self.cache = cache
        self.memory = memory
        self.router = router
        self.verifier = verifier
        self.tracer = tracer
        self.pii_redactor = pii_redactor
        self.on_guardrail_fail = on_guardrail_fail
        self.min_confidence = min_confidence

        # Cumulative stats
        self._query_count = 0
        self._cache_hits = 0
        self._guardrail_blocks = 0
        self._total_latency_ms = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def query(self, question: str, **kwargs: Any) -> SmartResult:
        """Execute the full smart pipeline."""
        start = time.perf_counter()
        self._query_count += 1

        # 1. Guardrails
        guardrail_checks = self._check_guardrails(question)
        if guardrail_checks.get("blocked"):
            self._guardrail_blocks += 1
            latency = (time.perf_counter() - start) * 1000
            self._total_latency_ms += latency
            return SmartResult(
                answer=guardrail_checks.get(
                    "message", "Query blocked by safety guardrails.",
                ),
                guardrail_checks=guardrail_checks,
                latency_ms=latency,
                metadata={"blocked": True},
            )

        # 2. PII redaction on input
        working_question = self._redact_pii(question)

        # 3. Cache lookup
        cached_answer = self._check_cache(working_question)
        if cached_answer is not None:
            self._cache_hits += 1
            latency = (time.perf_counter() - start) * 1000
            self._total_latency_ms += latency
            return SmartResult(
                answer=cached_answer,
                confidence=1.0,
                cached=True,
                guardrail_checks=guardrail_checks,
                latency_ms=latency,
            )

        # 4. Contextualise with memory
        contextualised = self._contextualize(working_question)

        # 5. Route and retrieve
        answer, sources, route_taken = self._route_and_retrieve(
            contextualised, **kwargs,
        )

        # 6. Verify answer
        verification: dict[str, Any] = {}
        if answer:
            verification = self._verify(answer, sources)
            v_confidence = verification.get("confidence", 1.0)
            if v_confidence < self.min_confidence:
                answer = (
                    "I'm not confident enough in this answer. "
                    + answer
                )

        # 7. PII redaction on output
        answer = self._redact_pii(answer)

        # 8. Store in cache
        self._store_cache(working_question, answer)

        # 9. Add to memory
        self._add_to_memory(question, answer)

        latency = (time.perf_counter() - start) * 1000
        self._total_latency_ms += latency

        confidence = verification.get("confidence", 0.5) if verification else 0.5
        return SmartResult(
            answer=answer,
            confidence=confidence,
            sources=sources,
            guardrail_checks=guardrail_checks,
            verification=verification,
            route_taken=route_taken,
            latency_ms=latency,
        )

    async def aquery(self, question: str, **kwargs: Any) -> SmartResult:
        """Async version of query."""
        return await asyncio.to_thread(self.query, question, **kwargs)

    # ── Orchestration steps ──────────────────────────────────────────────

    def _check_guardrails(self, question: str) -> dict[str, Any]:
        """Run all guardrails and return a summary dict."""
        results: dict[str, Any] = {"passed": True, "checks": []}
        for guardrail in self.guardrails:
            try:
                check = self._run_guardrail(guardrail, question)
                results["checks"].append(check)
                if not check.get("passed", True):
                    results["passed"] = False
                    if self.on_guardrail_fail == "block":
                        results["blocked"] = True
                        results["message"] = check.get(
                            "message", "Query blocked by safety guardrails.",
                        )
                        return results
            except Exception as exc:
                results["checks"].append({"error": str(exc), "passed": True})
        return results

    @staticmethod
    def _run_guardrail(guardrail: Any, question: str) -> dict[str, Any]:
        """Execute a single guardrail, adapting to its interface."""
        # Support objects with .check(), .scan(), or direct __call__
        if hasattr(guardrail, "check"):
            result = guardrail.check(question)
        elif hasattr(guardrail, "scan"):
            result = guardrail.scan(question)
        elif callable(guardrail):
            result = guardrail(question)
        else:
            return {"passed": True, "note": "Unsupported guardrail interface"}

        # Normalise result to dict
        if isinstance(result, dict):
            return result
        if isinstance(result, bool):
            return {"passed": result}
        if hasattr(result, "is_safe"):
            return {"passed": result.is_safe, "detail": str(result)}
        return {"passed": True, "raw": str(result)}

    def _check_cache(self, question: str) -> Optional[str]:
        """Look up the question in the semantic cache."""
        if self.cache is None:
            return None
        try:
            if hasattr(self.cache, "get"):
                hit = self.cache.get(question)
            elif hasattr(self.cache, "lookup"):
                hit = self.cache.lookup(question)
            else:
                return None
            # A cache miss is typically None or empty string
            if hit:
                return str(hit)
        except Exception:
            pass
        return None

    def _contextualize(self, question: str) -> str:
        """Contextualise the question with conversation memory."""
        if self.memory is None:
            return question
        try:
            if hasattr(self.memory, "contextualize"):
                return self.memory.contextualize(question)
            if hasattr(self.memory, "add_context"):
                return self.memory.add_context(question)
        except Exception:
            pass
        return question

    def _route_and_retrieve(
        self, question: str, **kwargs: Any,
    ) -> tuple[str, list, str]:
        """Route the query and produce an answer + sources."""
        route_taken = "pipeline"

        # If a router is available, let it decide
        if self.router is not None:
            try:
                if hasattr(self.router, "query"):
                    result = self.router.query(question, **kwargs)
                    answer = result.answer if hasattr(result, "answer") else str(result)
                    sources = result.metadata.get("sources", []) if hasattr(result, "metadata") else []
                    route_taken = (
                        result.metadata.get("route", "router")
                        if hasattr(result, "metadata") else "router"
                    )
                    return answer, sources, route_taken
                elif hasattr(self.router, "route"):
                    decision = self.router.route(question)
                    route_taken = str(decision.route.value) if hasattr(decision, "route") else "router"
            except Exception:
                pass

        # Fallback to pipeline
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, "query"):
                    result = self.pipeline.query(question, **kwargs)
                    answer = result.answer if hasattr(result, "answer") else str(result)
                    sources = []
                    if hasattr(result, "sources"):
                        sources = result.sources
                    elif hasattr(result, "results"):
                        sources = result.results
                    return answer, sources, route_taken
                elif hasattr(self.pipeline, "run"):
                    result = self.pipeline.run(question, **kwargs)
                    return str(result), [], route_taken
            except Exception:
                pass

        return "", [], "none"

    def _verify(self, answer: str, sources: list) -> dict[str, Any]:
        """Verify the answer using the configured verifier."""
        if self.verifier is None:
            return {}
        try:
            if hasattr(self.verifier, "verify"):
                result = self.verifier.verify(answer, sources)
            elif hasattr(self.verifier, "check"):
                result = self.verifier.check(answer, sources)
            elif callable(self.verifier):
                result = self.verifier(answer, sources)
            else:
                return {}

            if isinstance(result, dict):
                return result
            if hasattr(result, "confidence"):
                return {
                    "confidence": result.confidence,
                    "verified": getattr(result, "verified", True),
                    "detail": str(result),
                }
            return {"raw": str(result)}
        except Exception as exc:
            return {"error": str(exc)}

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text if a pii_redactor is configured."""
        if self.pii_redactor is None or not text:
            return text
        try:
            if hasattr(self.pii_redactor, "redact"):
                return self.pii_redactor.redact(text)
            if callable(self.pii_redactor):
                return self.pii_redactor(text)
        except Exception:
            pass
        return text

    def _store_cache(self, question: str, answer: str) -> None:
        """Store the question–answer pair in the cache."""
        if self.cache is None or not answer:
            return
        try:
            if hasattr(self.cache, "set"):
                self.cache.set(question, answer)
            elif hasattr(self.cache, "store"):
                self.cache.store(question, answer)
            elif hasattr(self.cache, "put"):
                self.cache.put(question, answer)
        except Exception:
            pass

    def _add_to_memory(self, question: str, answer: str) -> None:
        """Record the exchange in conversation memory."""
        if self.memory is None:
            return
        try:
            if hasattr(self.memory, "add"):
                self.memory.add(question=question, answer=answer)
            elif hasattr(self.memory, "append"):
                self.memory.append({"question": question, "answer": answer})
        except Exception:
            pass

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Cumulative pipeline statistics."""
        return {
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._query_count
                if self._query_count > 0 else 0.0
            ),
            "guardrail_blocks": self._guardrail_blocks,
            "avg_latency_ms": (
                self._total_latency_ms / self._query_count
                if self._query_count > 0 else 0.0
            ),
        }
