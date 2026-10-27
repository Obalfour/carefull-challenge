"""LLM provider abstraction (the provider-interface pattern from my course notes).

`generate_structured` validates the answer against a Pydantic schema, re-prompts with the
validation error, retries transient API errors, records tokens/latency/cost per call and
caches validated answers on disk.
"""
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .config import LLM_MAX_RPM, PRICING_USD_PER_MTOK

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """The model did not return a valid answer after all retries."""


class QuotaExhaustedError(Exception):
    """A daily quota is used up: retrying is pointless and the run should stop."""


class TransientLLMError(Exception):
    """An error worth retrying: rate limit, overloaded server, timeout."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.thinking_tokens += other.thinking_tokens


@dataclass
class CallRecord:
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_s: float
    cost_usd: float
    attempts: int
    cached: bool
    ok: bool
    error: str | None = None


def estimate_cost(model: str, usage: Usage) -> float:
    price = PRICING_USD_PER_MTOK.get(model)
    if price is None:
        return float("nan")
    billed_output = usage.output_tokens + usage.thinking_tokens
    return (usage.input_tokens * price["input"] + billed_output * price["output"]) / 1_000_000


class ResponseCache:
    """Thread-safe JSON file cache of validated responses, keyed by model + prompt."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        # Persist on every write so an interrupted run keeps everything it already paid for.
        with self._lock:
            self._data[key] = value
            self._write()

    def save(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


class RateLimiter:
    """Client-side sliding-window limit on requests per minute (0 = no limit)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._times: list[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.max_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 60]
                if len(self._times) < self.max_per_minute:
                    self._times.append(now)
                    return
                sleep_for = 60 - (now - self._times[0]) + 0.05
            time.sleep(sleep_for)


def _strip_fences(text: str) -> str:
    """Remove ```json fences in case a model wraps its JSON in markdown."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


class LLMProvider(ABC):
    """Common interface for any model provider."""

    def __init__(self, model_id: str, cache: ResponseCache | None = None, max_retries: int = 6,
                 max_rpm: int = 0):
        self.model_id = model_id
        self.cache = cache
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(max_rpm)
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()

    @abstractmethod
    def _call(self, prompt: str, system: str | None, schema_model: type[BaseModel],
              temperature: float, max_tokens: int) -> tuple[str, Usage]:
        """One raw API call. Must raise TransientLLMError for retryable failures."""

    def _record(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def generate_structured(self, prompt: str, schema_model: type[BaseModel], system: str | None = None,
                            purpose: str = "", temperature: float = 0.0, max_tokens: int = 4096) -> BaseModel:
        cache_key = ResponseCache.key(self.model_id, schema_model.__name__, system or "", prompt)
        if self.cache and (hit := self.cache.get(cache_key)):
            # Cached calls keep the metrics of the original call, so cost/latency stay comparable.
            self._record(CallRecord(**{**hit["record"], "cached": True}))
            return schema_model.model_validate(hit["response"])

        usage = Usage()
        api_time = 0.0
        last_error = None
        instruction = prompt
        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()
            start = time.perf_counter()
            try:
                text, call_usage = self._call(instruction, system, schema_model, temperature, max_tokens)
            except TransientLLMError as err:
                api_time += time.perf_counter() - start
                last_error = str(err)
                # Honor the server's suggested delay when it gives one (Gemini does on 429).
                wait = min(err.retry_after or 2 ** attempt, 90) + random.uniform(0, 1)
                log.warning("%s transient error (%s), retrying in %.1fs", self.model_id, err, wait)
                time.sleep(wait)
                continue
            except Exception as err:
                self._record(self._make_record(purpose, usage, api_time, attempt, ok=False, error=repr(err)))
                raise
            api_time += time.perf_counter() - start
            usage.add(call_usage)
            try:
                parsed = schema_model.model_validate_json(_strip_fences(text or ""))
            except ValidationError as err:
                last_error = f"schema validation failed: {err}"
                instruction = (
                    f"{prompt}\n\nYour previous answer did not match the required JSON schema. "
                    f"Error: {err}\nReturn ONLY the corrected JSON."
                )
                continue

            record = self._make_record(purpose, usage, api_time, attempt, ok=True)
            self._record(record)
            if self.cache:
                self.cache.set(cache_key, {"response": parsed.model_dump(), "record": asdict(record)})
            return parsed

        self._record(self._make_record(purpose, usage, api_time, self.max_retries, ok=False, error=last_error))
        raise LLMError(f"{self.model_id}: no valid answer after {self.max_retries} attempts: {last_error}")

    def _make_record(self, purpose, usage, latency, attempts, ok, error=None) -> CallRecord:
        return CallRecord(
            model=self.model_id, purpose=purpose,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens, latency_s=round(latency, 3),
            cost_usd=estimate_cost(self.model_id, usage), attempts=attempts,
            cached=False, ok=ok, error=error,
        )


class GoogleProvider(LLMProvider):
    """Gemini via google-genai, using native JSON-schema structured output."""

    def __init__(self, model_id: str, cache: ResponseCache | None = None, timeout_s: int = 90, **kwargs):
        super().__init__(model_id, cache, **kwargs)
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        )

    def _call(self, prompt, system, schema_model, temperature, max_tokens):
        import httpx
        from google.genai import errors

        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=schema_model,
        )
        try:
            resp = self.client.models.generate_content(model=self.model_id, contents=prompt, config=config)
        except errors.APIError as err:
            if err.code == 429 and "PerDay" in str(err):
                raise QuotaExhaustedError(f"{self.model_id}: daily request quota exhausted") from err
            if err.code in RETRYABLE_STATUS:
                hint = re.search(r"retry in ([\d.]+)s", str(err.message))
                raise TransientLLMError(f"HTTP {err.code}", float(hint.group(1)) if hint else None) from err
            raise
        except (httpx.TimeoutException, httpx.TransportError) as err:
            raise TransientLLMError(repr(err)) from err

        meta = resp.usage_metadata
        usage = Usage(
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
            thinking_tokens=(meta.thoughts_token_count or 0) if meta else 0,
        )
        return resp.text, usage


def get_provider(model_id: str, cache: ResponseCache | None = None) -> LLMProvider:
    """Factory: pick the provider from the model name. Add other providers here."""
    if model_id.startswith("gemini"):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is missing in .env")
        return GoogleProvider(model_id, cache, max_rpm=LLM_MAX_RPM)
    raise ValueError(f"No provider configured for model '{model_id}'")
