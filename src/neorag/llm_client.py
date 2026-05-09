"""LLM client wrapper for OpenJaw."""

# Original TODO: Implement LLM client wrapper with support for multiple providers
# - OpenAI API client
# - Anthropic API client
# - Ollama local client
# - Unified interface for chat completions
# - Tool calling support
# - Error handling and retries

import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

import aiohttp
from ..config import Config, ProviderConfig, ModelConfig, get_model_by_name_or_alias
from ..logging import get_logger


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
    tool_calls: List[ToolCall]
    model: str
    usage: Optional[Dict[str, Any]] = None


class LLMClientError(Exception):
    """LLM client related errors."""

    pass


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.session = None

        # Get default provider and model
        self.default_provider_name = config.llm.default_provider
        self.default_model_name = config.llm.default_model

        if self.default_provider_name not in config.llm.providers:
            raise LLMClientError(f"Default provider '{self.default_provider_name}' not found")

        self.default_provider = config.llm.providers[self.default_provider_name]
        self.default_model = get_model_by_name_or_alias(self.default_provider, self.default_model_name)

        if not self.default_model:
            raise LLMClientError(
                f"Default model '{self.default_model_name}' not found in provider '{self.default_provider_name}'"
            )

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions for function calling
            provider_name: Override default provider
            model_name: Override default model

        Returns:
            LLMResponse with content and any tool calls
        """
        if not self.session:
            raise LLMClientError("LLM client not initialized. Use async context manager.")

        # Determine provider and model
        provider_name = provider_name or self.default_provider_name
        model_name = model_name or self.default_model_name

        if provider_name not in self.config.llm.providers:
            raise LLMClientError(f"Provider '{provider_name}' not found")

        provider = self.config.llm.providers[provider_name]
        model = get_model_by_name_or_alias(provider, model_name)

        if not model:
            raise LLMClientError(f"Model '{model_name}' not found in provider '{provider_name}'")

        self.logger.debug(f"Making LLM request to {provider_name}/{model.name}")

        # Route to appropriate provider implementation
        if provider_name == "anthropic":
            return await self._chat_anthropic(messages, tools, provider, model)
        elif provider_name == "openai":
            return await self._chat_openai(messages, tools, provider, model)
        elif provider_name == "openrouter":
            return await self._chat_openai(messages, tools, provider, model)
        elif provider_name == "ollama":
            return await self._chat_ollama(messages, tools, provider, model)
        else:
            raise LLMClientError(f"Unsupported provider: {provider_name}")

    async def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> LLMResponse:
        """Handle Anthropic API requests."""
        headers = {"Content-Type": "application/json", "x-api-key": provider.api_key, "anthropic-version": "2023-06-01"}

        payload = {"model": model.name, "max_tokens": model.max_tokens, "messages": messages}

        if tools:
            payload["tools"] = tools

        try:
            async with self.session.post(f"{provider.base_url}/v1/messages", headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMClientError(f"Anthropic API error {response.status}: {error_text}")

                data = await response.json()
                return self._parse_anthropic_response(data, model.name)

        except aiohttp.ClientError as e:
            raise LLMClientError(f"Network error calling Anthropic API: {e}")

    async def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> LLMResponse:
        """Handle OpenAI API requests."""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {provider.api_key}"}

        payload = {"model": model.name, "messages": messages, "max_tokens": model.max_tokens}

        if tools:
            payload["tools"] = [{"type": "function", "function": tool} for tool in tools]

        try:
            async with self.session.post(
                f"{provider.base_url}/v1/chat/completions", headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMClientError(f"OpenAI API error {response.status}: {error_text}")

                data = await response.json()
                self.logger.debug(data)
                return self._parse_openai_response(data, model.name)

        except aiohttp.ClientError as e:
            raise LLMClientError(f"Network error calling OpenAI API: {e}")

    async def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> LLMResponse:
        """Handle Ollama API requests."""
        headers = {"Content-Type": "application/json"}

        payload = {"model": model.name, "messages": messages, "stream": False}

        # Note: Ollama tool support may vary by model
        if tools:
            payload["tools"] = tools

        self.logger.debug(f"Ollama API request payload: {json.dumps(payload, indent=2)}")

        try:
            async with self.session.post(f"{provider.base_url}/api/chat", headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMClientError(f"Ollama API error {response.status}: {error_text}")

                data = await response.json()
                self.logger.debug(f"Raw Ollama response data: {data}")
                return self._parse_ollama_response(data, model.name)

        except aiohttp.ClientError as e:
            raise LLMClientError(f"Network error calling Ollama API: {e}")

    def _parse_anthropic_response(self, data: Dict[str, Any], model_name: str) -> LLMResponse:
        """Parse Anthropic API response."""
        content = ""
        tool_calls = []

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
        tool_calls = []

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
        tool_calls = []

        # Parse Ollama tool calls - they have a different structure than OpenAI
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                # Extract tool name and arguments from Ollama's structure
                function_info = tool_call.get("function", {})
                tool_name = function_info.get("name", "")
                tool_args = function_info.get("arguments", {})

                # Skip tool calls with empty names (indicates parsing issues)
                if not tool_name:
                    self.logger.warning(f"Skipping tool call with empty name: {tool_call}")
                    continue

                # Handle arguments - they might be a string that needs JSON parsing
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse tool arguments as JSON: {tool_args}")
                        tool_args = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call.get("id", ""),
                        name=tool_name,
                        arguments=tool_args,
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls, model=model_name, usage=data.get("usage"))
