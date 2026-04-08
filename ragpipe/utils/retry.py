"""Retry/backoff utilities for HTTP calls and LLM providers.

Provides both sync and async retry decorators with exponential backoff,
configurable max attempts, jitter, and retryable exception filtering.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Type

logger = logging.getLogger(__name__)

# Default retryable exceptions — covers httpx, urllib, and generic network errors
_DEFAULT_RETRYABLE: tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    import httpx
    _DEFAULT_RETRYABLE = _DEFAULT_RETRYABLE + (
        httpx.HTTPStatusError,
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
    )
except ImportError:
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff (delay *= base each attempt).
        jitter: If True, add random jitter to delay (±25%).
        retryable_exceptions: Tuple of exception types that should trigger a retry.
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[Type[Exception], ...] = field(
        default_factory=lambda: _DEFAULT_RETRYABLE
    )

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for the given attempt number (0-indexed)."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = delay * (0.75 + random.random() * 0.5)
        return delay

    def is_retryable(self, exc: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        if isinstance(exc, self.retryable_exceptions):
            # For HTTP status errors, only retry on 429 and 5xx
            if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                code = exc.response.status_code
                return code == 429 or code >= 500
            return True
        return False


# Global default config — can be overridden per-call or per-module
DEFAULT_RETRY_CONFIG = RetryConfig()


def retry(
    fn: Callable | None = None,
    *,
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
) -> Callable:
    """Sync retry decorator with exponential backoff.

    Usage:
        @retry
        def call_api(): ...

        @retry(max_attempts=5, base_delay=0.5)
        def call_api(): ...

        @retry(config=RetryConfig(max_attempts=5))
        def call_api(): ...
    """
    def decorator(func: Callable) -> Callable:
        cfg = config or RetryConfig(
            max_attempts=max_attempts or DEFAULT_RETRY_CONFIG.max_attempts,
            base_delay=base_delay or DEFAULT_RETRY_CONFIG.base_delay,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(cfg.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not cfg.is_retryable(e) or attempt == cfg.max_attempts - 1:
                        raise
                    delay = cfg.compute_delay(attempt)
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt + 1, cfg.max_attempts, func.__name__, delay, e,
                    )
                    time.sleep(delay)
            raise last_exc  # pragma: no cover
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def aretry(
    fn: Callable | None = None,
    *,
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
) -> Callable:
    """Async retry decorator with exponential backoff.

    Usage:
        @aretry
        async def call_api(): ...

        @aretry(max_attempts=5)
        async def call_api(): ...
    """
    def decorator(func: Callable) -> Callable:
        cfg = config or RetryConfig(
            max_attempts=max_attempts or DEFAULT_RETRY_CONFIG.max_attempts,
            base_delay=base_delay or DEFAULT_RETRY_CONFIG.base_delay,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(cfg.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not cfg.is_retryable(e) or attempt == cfg.max_attempts - 1:
                        raise
                    delay = cfg.compute_delay(attempt)
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt + 1, cfg.max_attempts, func.__name__, delay, e,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # pragma: no cover
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def retry_call(
    fn: Callable,
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Call a function with retry logic (non-decorator usage).

    Usage:
        result = retry_call(requests.get, "https://api.example.com/data")
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    last_exc: Exception | None = None
    for attempt in range(cfg.max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not cfg.is_retryable(e) or attempt == cfg.max_attempts - 1:
                raise
            delay = cfg.compute_delay(attempt)
            logger.warning(
                "Retry %d/%d after %.2fs: %s", attempt + 1, cfg.max_attempts, delay, e,
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover


async def aretry_call(
    fn: Callable,
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Async call with retry logic (non-decorator usage)."""
    cfg = config or DEFAULT_RETRY_CONFIG
    last_exc: Exception | None = None
    for attempt in range(cfg.max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not cfg.is_retryable(e) or attempt == cfg.max_attempts - 1:
                raise
            delay = cfg.compute_delay(attempt)
            logger.warning(
                "Retry %d/%d after %.2fs: %s", attempt + 1, cfg.max_attempts, delay, e,
            )
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover
