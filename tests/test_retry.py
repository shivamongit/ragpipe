"""Tests for ragpipe.utils.retry — retry/backoff policies."""

import time
import pytest
from ragpipe.utils.retry import retry, aretry, RetryConfig, retry_call


class _TransientError(ConnectionError):
    pass


class _PermanentError(ValueError):
    pass


# ── RetryConfig ───────────────────────────────────────────────────────────────

def test_retry_config_defaults():
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.base_delay == 1.0
    assert cfg.exponential_base == 2.0
    assert cfg.jitter is True


def test_retry_config_compute_delay():
    cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
    assert cfg.compute_delay(0) == 1.0
    assert cfg.compute_delay(1) == 2.0
    assert cfg.compute_delay(2) == 4.0


def test_retry_config_max_delay():
    cfg = RetryConfig(base_delay=10.0, max_delay=15.0, jitter=False)
    assert cfg.compute_delay(5) == 15.0


def test_retry_config_jitter():
    cfg = RetryConfig(base_delay=1.0, jitter=True)
    delays = [cfg.compute_delay(0) for _ in range(20)]
    assert not all(d == delays[0] for d in delays), "Jitter should produce varying delays"


def test_retry_config_is_retryable():
    cfg = RetryConfig()
    assert cfg.is_retryable(ConnectionError("fail"))
    assert cfg.is_retryable(TimeoutError("timeout"))
    assert not cfg.is_retryable(ValueError("bad value"))


# ── @retry decorator ─────────────────────────────────────────────────────────

def test_retry_succeeds_first_try():
    call_count = 0

    @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    assert fn() == "ok"
    assert call_count == 1


def test_retry_succeeds_after_transient():
    call_count = 0

    @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert fn() == "recovered"
    assert call_count == 3


def test_retry_exhausts_attempts():
    @retry(config=RetryConfig(max_attempts=2, base_delay=0.01))
    def fn():
        raise ConnectionError("always fails")

    with pytest.raises(ConnectionError):
        fn()


def test_retry_no_retry_on_permanent_error():
    call_count = 0

    @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        fn()
    assert call_count == 1


def test_retry_with_kwargs():
    @retry(max_attempts=2, base_delay=0.01)
    def fn():
        return 42

    assert fn() == 42


# ── @aretry decorator ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_aretry_succeeds():
    @aretry(config=RetryConfig(max_attempts=3, base_delay=0.01))
    async def fn():
        return "async_ok"

    assert await fn() == "async_ok"


@pytest.mark.asyncio
async def test_aretry_retries_transient():
    call_count = 0

    @aretry(config=RetryConfig(max_attempts=3, base_delay=0.01))
    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("transient")
        return "recovered"

    assert await fn() == "recovered"
    assert call_count == 2


# ── retry_call ────────────────────────────────────────────────────────────────

def test_retry_call_success():
    result = retry_call(lambda: 99, config=RetryConfig(max_attempts=1))
    assert result == 99


def test_retry_call_with_transient():
    count = {"n": 0}

    def flaky():
        count["n"] += 1
        if count["n"] < 2:
            raise ConnectionError("fail")
        return "ok"

    result = retry_call(flaky, config=RetryConfig(max_attempts=3, base_delay=0.01))
    assert result == "ok"
