from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

ModelRole = Literal["chat", "deep", "utility"]

_MODEL_SETTING_BY_ROLE: dict[ModelRole, str] = {
    "chat": "llm_model",
    "deep": "llm_deep_model",
    "utility": "llm_utility_model",
}
_TEST_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "请只回复“连接正常”。"}]
_URL_QUERY_RE = re.compile(r"\b(https?://[^\s?#]+)\?[^\s#]*", re.IGNORECASE)
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(api[ _-]?key|access[ _-]?token|refresh[ _-]?token|authorization|token)"
    r"(\s*(?:=|:)\s*|\s+)(?:bearer\s+)?['\"]?[^\s,'\";)}]+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_KEY_SHAPE_RE = re.compile(r"\b(?:sk|rk|pk|api)[-_][A-Za-z0-9._-]{8,}\b", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\b")
_LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9_+/=-]{40,}\b")


class LLMConfigurationError(ValueError):
    pass


class LLMProviderError(RuntimeError):
    """A provider error whose public message has had credentials removed."""


@dataclass(frozen=True, slots=True)
class ModelResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    model: str
    usage_estimated: bool = True
    tool_calls: tuple[dict[str, Any], ...] | None = None


def sanitize_error(error: BaseException | str, *, secrets: Iterable[str] = ()) -> str:
    """Return an error suitable for a UI or log without keys or URL queries."""

    message = str(error) or type(error).__name__
    for secret in sorted({str(value) for value in secrets if value}, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    message = _URL_QUERY_RE.sub(r"\1", message)
    message = _NAMED_SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", message)
    message = _BEARER_RE.sub("Bearer [REDACTED]", message)
    message = _KEY_SHAPE_RE.sub("[REDACTED]", message)
    message = _JWT_RE.sub("[REDACTED]", message)
    message = _LONG_SECRET_RE.sub("[REDACTED]", message)
    return message


def _estimate_tokens(value: Any) -> int:
    """Conservative local estimate that works for both CJK and Latin text."""

    if isinstance(value, Mapping):
        text = " ".join(f"{key} {_content_text(item)}" for key, item in value.items())
    elif isinstance(value, list):
        text = " ".join(_content_text(item) for item in value)
    else:
        text = str(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def _messages_tokens(messages: list[dict[str, Any]]) -> int:
    # Include a small per-message allowance for role and chat framing.
    return sum(_estimate_tokens(message) + 4 for message in messages)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        if "text" in content:
            return str(content["text"])
        return " ".join(_content_text(value) for value in content.values())
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    text = getattr(content, "text", None)
    return str(text) if text is not None else str(content)


def _usage_value(usage: Any, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


class LLMBackend(ABC):
    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        raise NotImplementedError

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        async for part in self.stream(messages):
            parts.append(part)
        return "".join(parts)

    async def complete_result(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        started = time.perf_counter()
        text = await self.complete(messages)
        return ModelResult(
            text=text,
            input_tokens=_messages_tokens(messages),
            output_tokens=_estimate_tokens(text),
            latency_ms=(time.perf_counter() - started) * 1000,
            provider=str(getattr(self, "provider", "unknown")),
            model=str(getattr(self, "model", "unknown")),
            usage_estimated=True,
        )


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
        model: str | None = None,
        client: Any = None,
    ):
        if client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": float(config["llm_timeout_seconds"]),
            }
            client = AsyncOpenAI(**kwargs)
        self.client = client
        self._api_key = api_key
        self.provider = str(config["llm_provider"])
        self.base_url = base_url
        self.model = model or str(config["llm_model"])
        self.temperature = float(config["llm_temperature"])
        self.max_tokens = int(config["llm_max_tokens"])

    def _token_limit(self) -> dict[str, int]:
        host = (urlsplit(self.base_url).hostname or "").lower()
        if host == "api.openai.com":
            return {"max_completion_tokens": self.max_tokens}
        return {"max_tokens": self.max_tokens}

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
                **self._token_limit(),
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except TimeoutError as exc:
            raise TimeoutError(sanitize_error(exc, secrets=(self._api_key,))) from None
        except Exception as exc:
            raise LLMProviderError(sanitize_error(exc, secrets=(self._api_key,))) from None

    async def complete_result(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            **self._token_limit(),
        }
        if tools is not None:
            kwargs["tools"] = tools
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except TimeoutError as exc:
            raise TimeoutError(sanitize_error(exc, secrets=(self._api_key,))) from None
        except Exception as exc:
            raise LLMProviderError(sanitize_error(exc, secrets=(self._api_key,))) from None
        choices = getattr(response, "choices", None) or []
        message = choices[0].message if choices else None
        text = _content_text(getattr(message, "content", None) if message else None)
        raw_tool_calls = getattr(message, "tool_calls", None) if message else None
        parsed_tool_calls: tuple[dict[str, Any], ...] | None = None
        if raw_tool_calls:
            parsed = []
            for tc in raw_tool_calls:
                func = getattr(tc, "function", None)
                parsed.append(
                    {
                        "id": str(getattr(tc, "id", "")),
                        "type": str(getattr(tc, "type", "function")),
                        "function": {
                            "name": str(getattr(func, "name", "") if func else ""),
                            "arguments": str(getattr(func, "arguments", "") if func else ""),
                        },
                    }
                )
            parsed_tool_calls = tuple(parsed)
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
        estimated = input_tokens is None or output_tokens is None
        return ModelResult(
            text=text,
            input_tokens=input_tokens if input_tokens is not None else _messages_tokens(messages),
            output_tokens=output_tokens if output_tokens is not None else _estimate_tokens(text),
            latency_ms=(time.perf_counter() - started) * 1000,
            provider=self.provider,
            model=self.model,
            usage_estimated=estimated,
            tool_calls=parsed_tool_calls,
        )


@dataclass(frozen=True, slots=True)
class _Connection:
    provider: str
    base_url: str
    api_key: str

    @property
    def key_fingerprint(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()


def _connection(config: Mapping[str, Any]) -> _Connection:
    provider = str(config["llm_provider"]).strip().lower()
    if provider != "openai":
        raise LLMConfigurationError("当前只支持 OpenAI 兼容接口，请在模型中心重新配置")
    base_url = str(config.get("openai_base_url") or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise LLMConfigurationError("OpenAI 兼容端点必须是有效的 HTTP(S) 地址")
    # The OpenAI SDK requires a non-empty client value even when a compatible
    # local service has authentication disabled.  The placeholder is never
    # persisted and is only sent to that explicitly configured endpoint.
    key = str(config.get("openai_api_key") or "") or "not-required"
    return _Connection(provider, base_url, key)


def _select_model(config: Mapping[str, Any], role: ModelRole) -> str:
    if role not in _MODEL_SETTING_BY_ROLE:
        raise LLMConfigurationError(f"不支持的模型角色：{role}")
    model = str(config.get(_MODEL_SETTING_BY_ROLE[role]) or config.get("llm_model") or "").strip()
    if not model:
        raise LLMConfigurationError("请先配置模型 ID")
    return model


class ModelGateway:
    """Reusable provider client cache and high-level model operations."""

    def __init__(
        self,
        *,
        openai_client_factory: Callable[..., Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._openai_client_factory = openai_client_factory
        self._monotonic = monotonic
        self._clients: dict[tuple[Any, ...], Any] = {}
        self._backends: dict[tuple[Any, ...], LLMBackend] = {}
        self._model_catalog: dict[tuple[str, str, str], tuple[float, tuple[str, ...]]] = {}
        self._catalog_lock = asyncio.Lock()
        self._closed = False

    def _cache_key(
        self, config: Mapping[str, Any], connection: _Connection, model: str
    ) -> tuple[Any, ...]:
        return (
            connection.provider,
            connection.base_url.rstrip("/"),
            connection.key_fingerprint,
            model,
            float(config["llm_temperature"]),
            int(config["llm_max_tokens"]),
            float(config["llm_timeout_seconds"]),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ModelGateway 已关闭")

    def _new_client(
        self, config: Mapping[str, Any], connection: _Connection, cache_key: tuple[Any, ...]
    ) -> Any:
        if cache_key in self._clients:
            return self._clients[cache_key]
        factory = self._openai_client_factory
        if factory is None:
            from openai import AsyncOpenAI

            factory = AsyncOpenAI
        kwargs: dict[str, Any] = {
            "api_key": connection.api_key,
            "base_url": connection.base_url,
            "timeout": float(config["llm_timeout_seconds"]),
        }
        client = factory(**kwargs)
        self._clients[cache_key] = client
        return client

    def build_backend(self, config: dict[str, Any], *, role: ModelRole = "chat") -> LLMBackend:
        self._ensure_open()
        connection = _connection(config)
        model = _select_model(config, role)
        cache_key = self._cache_key(config, connection, model)
        cached = self._backends.get(cache_key)
        if cached is not None:
            return cached
        client = self._new_client(config, connection, cache_key)
        backend: LLMBackend = OpenAICompatibleBackend(
            config,
            api_key=connection.api_key,
            base_url=connection.base_url,
            model=model,
            client=client,
        )
        self._backends[cache_key] = backend
        return backend

    async def complete(
        self,
        config: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        role: ModelRole = "chat",
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        return await self.build_backend(config, role=role).complete_result(messages, tools=tools)

    async def stream(
        self,
        config: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        role: ModelRole = "chat",
    ) -> AsyncIterator[str]:
        backend = self.build_backend(config, role=role)
        async for part in backend.stream(messages):
            yield part

    async def list_models(self, config: dict[str, Any], *, force: bool = False) -> list[str]:
        self._ensure_open()
        connection = _connection(config)
        catalog_key = (
            connection.provider,
            connection.base_url.rstrip("/"),
            connection.key_fingerprint,
        )
        ttl = max(0.0, float(config.get("model_catalog_cache_minutes", 10))) * 60
        now = self._monotonic()
        cached = self._model_catalog.get(catalog_key)
        if not force and cached is not None and now - cached[0] < ttl:
            return list(cached[1])

        async with self._catalog_lock:
            now = self._monotonic()
            cached = self._model_catalog.get(catalog_key)
            if not force and cached is not None and now - cached[0] < ttl:
                return list(cached[1])
            # Discovery does not require a selected chat model yet.
            model = str(config.get("llm_model") or "<model-catalog>")
            cache_key = self._cache_key(config, connection, model)
            client = self._new_client(config, connection, cache_key)
            try:
                page = await _maybe_await(client.models.list())
                ids = sorted(set(await _model_ids(page)))
            except Exception as exc:
                safe = sanitize_error(exc, secrets=(connection.api_key,))
                raise LLMProviderError(safe) from None
            self._model_catalog[catalog_key] = (self._monotonic(), tuple(ids))
            return ids

    async def test_model(
        self,
        config: dict[str, Any],
        *,
        role: ModelRole = "chat",
        timeout_seconds: float | None = None,
    ) -> ModelResult:
        self._ensure_open()
        connection = _connection(config)
        timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(config["llm_timeout_seconds"])
        )
        try:
            return await asyncio.wait_for(
                self.complete(config, [dict(message) for message in _TEST_MESSAGES], role=role),
                timeout=timeout,
            )
        except TimeoutError:
            raise TimeoutError(f"模型测试超时（{timeout:g} 秒）") from None
        except (LLMConfigurationError, LLMProviderError):
            raise
        except Exception as exc:
            safe = sanitize_error(exc, secrets=(connection.api_key,))
            raise LLMProviderError(safe) from None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        clients = list({id(client): client for client in self._clients.values()}.values())
        self._clients.clear()
        self._backends.clear()
        self._model_catalog.clear()
        for client in clients:
            closer = getattr(client, "close", None) or getattr(client, "aclose", None)
            if closer is None:
                continue
            result = closer()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> ModelGateway:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def aclose(self) -> None:
        await self.close()


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _model_ids(page: Any) -> list[str]:
    if hasattr(page, "__aiter__"):
        items = [item async for item in page]
    elif isinstance(page, Mapping):
        items = page.get("data") or page.get("models") or []
    elif isinstance(page, (list, tuple, set)):
        items = page
    else:
        items = getattr(page, "data", None) or getattr(page, "models", None) or []
    ids: list[str] = []
    for item in items:
        value = item.get("id") if isinstance(item, Mapping) else getattr(item, "id", item)
        model_id = str(value or "").strip()
        if model_id:
            ids.append(model_id)
    return ids


_DEFAULT_GATEWAY = ModelGateway()


def build_backend(config: dict[str, Any], *, role: ModelRole = "chat") -> LLMBackend:
    return _DEFAULT_GATEWAY.build_backend(config, role=role)


async def list_models(
    config: dict[str, Any], force: bool = False, *, gateway: ModelGateway | None = None
) -> list[str]:
    return await (gateway or _DEFAULT_GATEWAY).list_models(config, force=force)


async def test_model(
    config: dict[str, Any],
    *,
    role: ModelRole = "chat",
    timeout_seconds: float | None = None,
    gateway: ModelGateway | None = None,
) -> ModelResult:
    return await (gateway or _DEFAULT_GATEWAY).test_model(
        config, role=role, timeout_seconds=timeout_seconds
    )


async def collect_with_timeout(
    backend: LLMBackend,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
) -> str:
    return await asyncio.wait_for(backend.complete(messages), timeout=timeout_seconds)
