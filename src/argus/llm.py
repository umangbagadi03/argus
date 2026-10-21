"""
Every LLM call in this project goes through `RateLimitedGemini`, never
through a raw `ChatGoogleGenerativeAI` instance. That's the single decision
that makes this project actually runnable on the free tier instead of
falling over after the fourth or fifth call.

Three things are stacked here, each solving a different failure mode:

1. Per-model rate limiting (`InMemoryRateLimiter`). Google enforces
   requests-per-minute quotas *per model*, not per API key — so
   `gemini-2.5-flash` and `gemini-2.5-flash-lite` have separate, independent
   budgets even on the same key. Rate-limiting client-side means the graph
   paces itself instead of firing requests and hoping, which is what turns
   "occasional 429s" into "no 429s".

2. Model fallback. If a model's local budget is exhausted anyway (a burst
   of parallel `Send` branches can outrun even correct client-side pacing),
   fall through to the next model in the chain rather than raising. Because
   quotas are per-model, this is a real second budget, not a fake retry
   against the same wall.

3. Disk caching, keyed on the exact prompt (+ schema, for structured calls).
   Retries, repeated demo runs, and the evaluation harness re-scoring the
   same benchmark questions should not cost quota twice for an identical
   call.

Free-tier numbers change without much notice (Google cut them 50-80% in
December 2025) — verify current values at
https://ai.google.dev/gemini-api/docs/rate-limits before relying on the
defaults below for anything beyond a demo.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from .cache import DiskCache
from .config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ModelTier:
    name: str
    rpm: int  # requests-per-minute quota Google grants this model on the free tier
    rpd: int  # requests-per-day quota (not enforced client-side here, just documentation)


# Reasonable, currently-verified defaults as of mid-2026: 2.5 Flash and
# Flash-Lite are both firmly established free-tier models. Pro is
# deliberately excluded from the default chain — 5 RPM / ~100 RPD is tight
# enough that a multi-agent graph doing 6-10+ calls per research session
# will exhaust it almost immediately. Swap in `gemini-2.5-pro` yourself (or
# newer 3.x-series models once they're out of preview) if you want higher
# per-call quality and can tolerate a much slower pipeline.
DEFAULT_FALLBACK_CHAIN = [
    ModelTier("gemini-2.5-flash", rpm=10, rpd=250),
    ModelTier("gemini-2.5-flash-lite", rpm=15, rpd=1000),
]


class InvalidAPIKeyError(RuntimeError):
    """Raised when Google rejects the configured key. Not worth falling back
    on: every model in the chain shares the same key, so they'd all fail."""


def _raise_if_auth_error(exc: Exception) -> None:
    msg = str(exc)
    if "API_KEY_INVALID" in msg or "API key not valid" in msg or "PERMISSION_DENIED" in msg:
        raise InvalidAPIKeyError(
            "Google rejected GOOGLE_API_KEY. Create a new key at "
            "https://aistudio.google.com/apikey and update your .env file."
        ) from exc


class RateLimitedGemini:
    """A drop-in-ish LLM handle exposing `.invoke(prompt) -> str` and
    `.invoke_structured(prompt, Schema) -> Schema`, backed by the fallback
    chain described above."""

    def __init__(
        self,
        tiers: Optional[list[ModelTier]] = None,
        cache: Optional[DiskCache] = None,
        temperature: float = 0.2,
    ):
        self.tiers = tiers or DEFAULT_FALLBACK_CHAIN
        self.cache = cache or DiskCache()
        self._models: dict[str, ChatGoogleGenerativeAI] = {}

        for tier in self.tiers:
            limiter = InMemoryRateLimiter(
                requests_per_second=tier.rpm / 60,
                check_every_n_seconds=0.5,
                # A small burst allowance (2) rather than 1: lets the very
                # first call of a run through immediately instead of always
                # waiting a full tick, without meaningfully risking the RPM
                # budget.
                max_bucket_size=2,
            )
            model = ChatGoogleGenerativeAI(
                model=tier.name,
                temperature=temperature,
                rate_limiter=limiter,
                google_api_key=settings.google_api_key,
            )
            self._models[tier.name] = model

    def _cache_key(self, prompt: str, schema_name: str = "") -> str:
        return hashlib.sha256(f"{schema_name}:{prompt}".encode("utf-8")).hexdigest()

    def invoke(self, prompt: str, **kwargs) -> str:
        key = self._cache_key(prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        last_err: Optional[Exception] = None
        for tier in self.tiers:
            try:
                result = self._models[tier.name].with_retry(
                    stop_after_attempt=3,
                    wait_exponential_jitter=True,
                ).invoke(prompt, **kwargs).content
                self.cache.set(key, result)
                return result
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
                _raise_if_auth_error(exc)
                logger.warning("Model %s failed (%s); falling back", tier.name, exc)
                last_err = exc
                continue
        raise RuntimeError(f"All models in the fallback chain were exhausted: {last_err}")

    def invoke_structured(self, prompt: str, schema: Type[T]) -> T:
        key = self._cache_key(prompt, schema.__name__)
        cached = self.cache.get(key)
        if cached is not None:
            return schema.model_validate_json(cached)

        last_err: Optional[Exception] = None
        for tier in self.tiers:
            try:
                structured_model = self._models[tier.name].with_structured_output(schema).with_retry(
                    stop_after_attempt=3,
                    wait_exponential_jitter=True,
                )
                result: T = structured_model.invoke(prompt)
                self.cache.set(key, result.model_dump_json())
                return result
            except Exception as exc:  # noqa: BLE001
                _raise_if_auth_error(exc)
                logger.warning("Model %s failed structured call (%s); falling back", tier.name, exc)
                last_err = exc
                continue
        raise RuntimeError(f"All models in the fallback chain were exhausted: {last_err}")
