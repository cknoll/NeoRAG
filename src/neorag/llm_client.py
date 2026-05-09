"""LLM client wrapper for NeoRAG.

Synchronous unified LLM client with support for multiple providers
(Anthropic, OpenAI, OpenRouter, Ollama) and tool calling. Adapted from
an async/aiohttp original to use synchronous ``httpx`` so it composes
naturally with the rest of the (synchronous) NeoRAG pipeline.

Configuration source
--------------------
Provider/model/base-url defaults are read from :mod:`neorag.config`.
API keys are read from a TOML file (default: ``config.toml`` in the
project root; override with the ``NEORAG_CONFIG_TOML`` env var). See
``config-example.toml`` for the expected layout.

Logging
-------
Per the improvement-plan §3 P0.1 transition, structured logging is
deferred; this module simply uses ``print(...)`` for diagnostics.

Backends
--------
This module also defines a minimal :class:`LLMBackend` ``Protocol`` and
a :class:`StubBackend` to support offline tests and CI without API
access (improvement-plan §3 P0.1).
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import httpx

from . import config as cfg


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Represents a response from the LLM."""

    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


class LLMClientError(Exception):
    """LLM client related errors."""

    pass


# ---------------------------------------------------------------------------
# Backend protocol + stub
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal protocol every NeoRAG LLM backend must implement.

    The validator / refinement loop only needs a synchronous ``chat`` call
    that takes a list of OpenAI-style messages and returns an
    :class:`LLMResponse`. Tool-calling support is optional; pass ``tools``
    as ``None`` if the caller does not need it.
    """

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse: ...


class StubBackend:
    """Offline canned-response backend.

    Used by tests and CI where no network / API key is available
    (improvement-plan §3 P0.1). Returns ``canned_response`` verbatim and
    ignores ``messages`` / ``tools`` (but records the last call for
    introspection in tests).
    """

    def __init__(
        self,
        canned_response: str = '{"answer": "stub"}',
        model: str = "stub",
    ):
        self.canned_response = canned_response
        self.model = model
        self.last_messages: Optional[List[Dict[str, str]]] = None
        self.last_tools: Optional[List[Dict[str, Any]]] = None
        self.call_count = 0

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        self.call_count += 1
        return LLMResponse(content=self.canned_response, tool_calls=[], model=self.model)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_api_key_from_toml(
    provider_name: str,
    toml_path: Optional[Path] = None,
) -> Optional[str]:
    """Return the API key for ``provider_name`` read from a TOML config file.

    Returns ``None`` if no key is configured (e.g. for ``ollama`` or
    ``stub`` providers, or if the TOML file does not exist). Raises
    :class:`LLMClientError` if the TOML file exists but is malformed.
    """
    field_name = cfg.LLM_API_KEY_TOML_FIELD.get(provider_name)
    if field_name is None:
        return None

    path = Path(toml_path) if toml_path is not None else cfg.CONFIG_TOML_PATH
    if not path.is_file():
        return None

    try:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise LLMClientError(f"Failed to read API key from {path}: {e}") from e

    return data.get(field_name)


# ---------------------------------------------------------------------------
# Synchronous multi-provider LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified synchronous LLM client supporting multiple providers.

    Implements the :class:`LLMBackend` protocol. Construct directly with
    explicit parameters, or use :meth:`from_config` to pick up defaults
    from :mod:`neorag.config` and the API key from ``config.toml``.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        verbose: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.verbose = verbose

        if provider not in {"anthropic", "openai", "openrouter", "ollama"}:
            raise LLMClientError(f"Unsupported provider: {provider}")

        if provider in {"anthropic", "openai", "openrouter"} and not api_key:
            raise LLMClientError(
                f"Provider '{provider}' requires an API key. "
                f"Set '{cfg.LLM_API_KEY_TOML_FIELD.get(provider, '<api_key>')}' "
                f"in {cfg.CONFIG_TOML_PATH}."
            )

        # Reusable HTTP client; closed in close()/__exit__.
        self._http = httpx.Client(timeout=timeout_s)

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_config(cls, **overrides: Any) -> "LLMClient":
        """Build an :class:`LLMClient` from :mod:`neorag.config` defaults.

        Any keyword argument overrides the corresponding config value.
        The API key is read from ``config.toml`` (see
        :func:`load_api_key_from_toml`).
        """
        provider = overrides.pop("provider", cfg.LLM_PROVIDER)
        model = overrides.pop("model", cfg.LLM_MODEL)
        base_url = overrides.pop("base_url", cfg.LLM_BASE_URL)
        max_tokens = overrides.pop("max_tokens", cfg.LLM_MAX_TOKENS)
        timeout_s = overrides.pop("timeout_s", cfg.LLM_TIMEOUT_S)
        api_key = overrides.pop("api_key", None)
        if api_key is None:
            api_key = load_api_key_from_toml(provider)

        if overrides:
            raise TypeError(f"Unexpected kwargs for LLMClient.from_config: {sorted(overrides)}")

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    # -- context manager / cleanup -------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- logging shim ---------------------------------------------------------

    def _log(self, msg: str) -> None:
        # Logging is intentionally minimal for now (improvement-plan note:
        # structured logging is deferred). Use stderr so it doesn't pollute
        # stdout-based CLI output.
        if self.verbose:
            print(f"[llm_client] {msg}", file=sys.stderr)

    # -- public API -----------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Send a chat completion request to the configured provider."""
        self._log(f"Making LLM request to {self.provider}/{self.model}")

        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools)
        elif self.provider in {"openai", "openrouter"}:
            return self._chat_openai(messages, tools)
        elif self.provider == "ollama":
            return self._chat_ollama(messages, tools)
        else:  # pragma: no cover - guarded in __init__
            raise LLMClientError(f"Unsupported provider: {self.provider}")

    # -- per-provider implementations ----------------------------------------

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> LLMResponse:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        try:
            response = self._http.post(
                f"{self.base_url}/v1/messages", headers=headers, json=payload
            )
        except httpx.HTTPError as e:
            raise LLMClientError(f"Network error calling Anthropic API: {e}") from e

        if response.status_code != 200:
            raise LLMClientError(
                f"Anthropic API error {response.status_code}: {response.text}"
            )

        return self._parse_anthropic_response(response.json(), self.model)

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> LLMResponse:
        """Handle OpenAI-compatible API requests (also used for OpenRouter)."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or ''}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": tool} for tool in tools]

        try:
            response = self._http.post(
                f"{self.base_url}/v1/chat/completions", headers=headers, json=payload
            )
        except httpx.HTTPError as e:
            raise LLMClientError(f"Network error calling OpenAI API: {e}") from e

        if response.status_code != 200:
            raise LLMClientError(
                f"OpenAI API error {response.status_code}: {response.text}"
            )

        data = response.json()
        self._log(f"response: {data}")
        return self._parse_openai_response(data, self.model)

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        # Note: Ollama tool support may vary by model
        if tools:
            payload["tools"] = tools

        self._log(f"Ollama API request payload: {json.dumps(payload, indent=2)}")

        try:
            response = self._http.post(
                f"{self.base_url}/api/chat", headers=headers, json=payload
            )
        except httpx.HTTPError as e:
            raise LLMClientError(f"Network error calling Ollama API: {e}") from e

        if response.status_code != 200:
            raise LLMClientError(
                f"Ollama API error {response.status_code}: {response.text}"
            )

        data = response.json()
        self._log(f"Raw Ollama response data: {data}")
        return self._parse_ollama_response(data, self.model)

    # -- response parsing -----------------------------------------------------

    def _parse_anthropic_response(self, data: Dict[str, Any], model_name: str) -> LLMResponse:
        """Parse Anthropic API response."""
        content = ""
        tool_calls: List[ToolCall] = []

        for content_block in data.get("content", []):
            if content_block.get("type") == "text":
                content += content_block.get("text", "")
            elif content_block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=content_block.get("id", ""),
                        name=content_block.get("name", ""),
                        arguments=content_block.get("input", {}),
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls, model=model_name, usage=data.get("usage"))

    def _parse_openai_response(self, data: Dict[str, Any], model_name: str) -> LLMResponse:
        """Parse OpenAI API response."""
        message = data["choices"][0]["message"]
        content = message.get("content", "") or ""
        tool_calls: List[ToolCall] = []

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:

                # preprocess tool_call_args (because llm might deliver them badly)
                tool_call_args = tool_call["function"]["arguments"]
                assert isinstance(tool_call_args, str)

                if tool_call_args.endswith("}") and tool_call_args.count("}") > tool_call_args.count("{"):
                    tool_call_args = "{{}".format(tool_call_args)

                tool_calls.append(
                    ToolCall(
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call_args),
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls, model=model_name, usage=data.get("usage"))

    def _parse_ollama_response(self, data: Dict[str, Any], model_name: str) -> LLMResponse:
        """Parse Ollama API response."""
        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls: List[ToolCall] = []

        # Parse Ollama tool calls - they have a different structure than OpenAI
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                # Extract tool name and arguments from Ollama's structure
                function_info = tool_call.get("function", {})
                tool_name = function_info.get("name", "")
                tool_args = function_info.get("arguments", {})

                # Skip tool calls with empty names (indicates parsing issues)
                if not tool_name:
                    self._log(f"Skipping tool call with empty name: {tool_call}")
                    continue

                # Handle arguments - they might be a string that needs JSON parsing
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        self._log(f"Failed to parse tool arguments as JSON: {tool_args}")
                        tool_args = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call.get("id", ""),
                        name=tool_name,
                        arguments=tool_args,
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls, model=model_name, usage=data.get("usage"))
