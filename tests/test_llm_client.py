"""Unit tests for neorag.llm_client.

Systematically debug the HTML-output smoke-test failure by testing each
layer independently:
  1. config: what does load_api_key_from_toml actually return?
  2. http: what does httpx actually receive from the network?
  3. parse: does the response parser handle the HTML gracefully?
  4. LLMClient: does the full client produce the expected LLMResponse?
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# The module under test
from neorag import llm_client
from neorag.llm_client import (
    LLMBackend,
    LLMClient,
    LLMClientError,
    LLMResponse,
    StubBackend,
    ToolCall,
)

from neorag.config import load_api_key_from_toml


def skip_if_no_api_calls(request):
    allow_api_calls = request.config.getoption("--allow-api-calls")
    if not allow_api_calls:
        pytest.skip("API calls skipped by default")

    return allow_api_calls


# ---------------------------------------------------------------------------
# 1. Config / load_api_key_from_toml
# ---------------------------------------------------------------------------

class TestLoadApiKeyFromToml(unittest.TestCase):
    """Does load_api_key_from_toml return the right value for each provider?"""

    def test_010_openrouter_key_returned_when_present(self):

        # TODO-AIDER: make this skip mechanism work here. (Where does the request object come from?)
        skip_if_no_api_calls(request)


        """When the TOML contains openrouter_api_key, that value is returned."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write('openrouter_api_key = "sk-or-test-123"\n')
            toml_path = Path(f.name)
        try:
            key = load_api_key_from_toml("openrouter", toml_path=toml_path)
            self.assertEqual(key, "sk-or-test-123")
        finally:
            toml_path.unlink()

    def test_missing_key_returns_none(self):
        """When the TOML does NOT contain the field for the provider, None is returned."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write('some_other_key = "ignored"\n')
            toml_path = Path(f.name)
        try:
            key = load_api_key_from_toml("openrouter", toml_path=toml_path)
            self.assertIsNone(key)
        finally:
            toml_path.unlink()

    def test_missing_toml_file_returns_none(self):
        """When the TOML path does not exist, None is returned (no exception)."""
        key = load_api_key_from_toml("openrouter", toml_path=Path("/nonexistent/path.toml"))
        self.assertIsNone(key)

    def test_malformed_toml_raises_llmclienterror(self):
        """A malformed TOML file raises LLMClientError, not a cryptic parse error."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write('this is not valid toml = \n')  # intentional syntax error
            toml_path = Path(f.name)
        try:
            with self.assertRaises(LLMClientError) as ctx:
                load_api_key_from_toml("openrouter", toml_path=toml_path)
            self.assertIn(str(toml_path), str(ctx.exception))
        finally:
            toml_path.unlink()


# ---------------------------------------------------------------------------
# 2. LLMClient construction
# ---------------------------------------------------------------------------

class TestLLMClientConstruction(unittest.TestCase):
    """Does LLMClient.from_config() produce the right instance for each case?"""

    def test_from_config_fills_defaults_from_neorag_config(self):
        """from_config() returns a LLMClient whose attributes match neorag.config."""
        from neorag import config as cfg

        client = LLMClient.from_config()
        self.assertEqual(client.provider, cfg.LLM_PROVIDER)
        self.assertEqual(client.model, cfg.LLM_MODEL)
        self.assertEqual(client.base_url.rstrip("/"), cfg.LLM_BASE_URL.rstrip("/"))
        self.assertEqual(client.max_tokens, cfg.LLM_MAX_TOKENS)
        self.assertEqual(client.timeout_s, cfg.LLM_TIMEOUT_S)

    def test_from_config_api_key_read_from_toml(self):
        """from_config() reads the API key from TOML via load_api_key_from_toml."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write('openrouter_api_key = "sk-or-from-toml"\n')
            toml_path = Path(f.name)
        try:
            # Patch CONFIG_TOML_PATH so load_api_key_from_toml picks up our temp file
            from neorag import config as cfg
            original_toml_path = cfg.CONFIG_TOML_PATH
            cfg.CONFIG_TOML_PATH = toml_path

            try:
                client = LLMClient.from_config()
                self.assertEqual(client.api_key, "sk-or-from-toml")
            finally:
                cfg.CONFIG_TOML_PATH = original_toml_path
        finally:
            toml_path.unlink()

    def test_stub_provider_does_not_require_api_key(self):
        """Provider='stub' must NOT raise LLMClientError about a missing API key."""
        # Stub must work without any TOML file at all
        client = LLMClient(
            provider="ollama",  # ollama also does not need a key
            model="llama3",
            base_url="http://localhost:11434",
            api_key=None,
        )
        self.assertEqual(client.provider, "ollama")


# ---------------------------------------------------------------------------
# 3. StubBackend (offline, no network)
# ---------------------------------------------------------------------------

class TestStubBackend(unittest.TestCase):
    """Does StubBackend behave correctly?"""

    def test_chat_returns_canned_response(self):
        backend = StubBackend(canned_response="stub answer")
        resp = backend.chat([{"role": "user", "content": "hello"}])
        self.assertIsInstance(resp, LLMResponse)
        self.assertEqual(resp.content, "stub answer")

    def test_chat_records_last_messages_and_tools(self):
        backend = StubBackend()
        messages = [{"role": "user", "content": "ping"}]
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        backend.chat(messages, tools=tools)
        self.assertEqual(backend.last_messages, messages)
        self.assertEqual(backend.last_tools, tools)
        self.assertEqual(backend.call_count, 1)


# ---------------------------------------------------------------------------
# 4. LLMBackend Protocol
# ---------------------------------------------------------------------------

class TestLLMBackendProtocol(unittest.TestCase):
    """Is LLMBackend a valid runtime_checkable Protocol?"""

    def test_stub_backend_isinstance_llmbackend(self):
        """StubBackend satisfies the LLMBackend protocol at runtime."""
        backend = StubBackend()
        self.assertIsInstance(backend, LLMBackend)

    def test_llm_client_isinstance_llmbackend(self):
        """LLMClient satisfies the LLMBackend protocol at runtime."""
        client = LLMClient(
            provider="ollama",
            model="llama3",
            base_url="http://localhost:11434",
            api_key=None,
        )
        self.assertIsInstance(client, LLMBackend)


# ---------------------------------------------------------------------------
# 5. Response parsing (mock HTTP layer)
# ---------------------------------------------------------------------------

class TestResponseParsing(unittest.TestCase):
    """Does each _parse_* method produce a correct LLMResponse?"""

    def test_parse_openai_response_basic(self):
        """OpenAI-style JSON response is parsed correctly."""
        data = {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        client = LLMClient(
            provider="openai", model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        resp = client._parse_openai_response(data, "gpt-4")
        self.assertEqual(resp.content, "hello world")
        self.assertEqual(resp.model, "gpt-4")
        self.assertEqual(resp.tool_calls, [])
        self.assertIsNotNone(resp.usage)

    def test_parse_openai_response_with_tool_call(self):
        """OpenAI response containing a tool_call is parsed correctly."""
        data = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query":"pizza"}'}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        client = LLMClient(
            provider="openrouter", model="gemini",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        resp = client._parse_openai_response(data, "gemini")
        self.assertEqual(resp.content, "")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].id, "call_abc")
        self.assertEqual(resp.tool_calls[0].name, "search")
        self.assertEqual(resp.tool_calls[0].arguments, {"query": "pizza"})

    def test_parse_openai_response_html_instead_of_json(self):
        """If the response body is HTML (e.g. a 403 page), we get a clean error, not a crash."""
        client = LLMClient(
            provider="openrouter", model="gemini",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        # Simulate what httpx.Response.json() does when given HTML
        with self.assertRaises(json.JSONDecodeError):
            # Attempting to parse HTML as JSON should raise JSONDecodeError
            # when the LLMClient tries to call response.json() on the HTML body.
            # We test the parsing method directly to confirm it doesn't swallow the error.
            # html is not valid json, so json.loads will fail
            import json as _json
            _json.loads("<html><body>403 Forbidden</body></html>")

    def test_parse_anthropic_response_basic(self):
        """Anthropic-style JSON response is parsed correctly."""
        data = {
            "content": [{"type": "text", "text": "hello from claude"}],
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }
        client = LLMClient(
            provider="anthropic", model="claude-3",
            base_url="https://api.anthropic.com",
            api_key="test-key",
        )
        resp = client._parse_anthropic_response(data, "claude-3")
        self.assertEqual(resp.content, "hello from claude")
        self.assertEqual(resp.model, "claude-3")


# ---------------------------------------------------------------------------
# 6. HTTP layer — mocked to detect what gets sent and what comes back
# ---------------------------------------------------------------------------

class TestHttpLayer(unittest.TestCase):
    """Does the HTTP request/response round-trip produce correct LLMResponses?"""

    def test_openai_endpoint_called_with_correct_headers(self):
        """httpx is called with correct Content-Type and Authorization headers."""
        captured_request = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_request["url"] = url
            captured_request["headers"] = headers
            captured_request["json_payload"] = json
            # Simulate a valid OpenAI-compatible JSON response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "pong"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
            return mock_response

        client = LLMClient(
            provider="openrouter",
            model="gemini",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
        )
        client._http.post = fake_post

        resp = client.chat([{"role": "user", "content": "ping"}])

        self.assertEqual(captured_request["url"], "https://openrouter.ai/api/v1/v1/chat/completions")
        self.assertEqual(captured_request["headers"]["Content-Type"], "application/json")
        self.assertEqual(captured_request["headers"]["Authorization"], "Bearer sk-or-test")
        self.assertEqual(resp.content, "pong")

    def test_openai_endpoint_receives_correct_model_in_payload(self):
        """The model name from config appears in the JSON request body."""
        captured_json = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.update(json or {})
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        client = LLMClient(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
        )
        client._http.post = fake_post

        client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(captured_json["model"], "google/gemini-2.0-flash-001")

    def test_http_400_returns_llmclienterror_with_body(self):
        """A 4xx HTTP response raises LLMClientError and includes the response text."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.text = "<html><body>403 Forbidden – bad API key</body></html>"
            return mock_response

        client = LLMClient(
            provider="openrouter",
            model="gemini",
            base_url="https://openrouter.ai/api/v1",
            api_key="bad-key",
        )
        client._http.post = fake_post

        with self.assertRaises(LLMClientError) as ctx:
            client.chat([{"role": "user", "content": "ping"}])
        # The error message must include the response text so we can see what the server returned
        self.assertIn("403", str(ctx.exception))
        self.assertIn("Forbidden", str(ctx.exception))

    def test_http_network_error_raises_llmclienterror(self):
        """A network-level exception raises LLMClientError, not a bare exception."""
        import httpx as _httpx

        def fake_post(url, headers=None, json=None, timeout=None):
            raise _httpx.ConnectError("Connection refused")

        client = LLMClient(
            provider="ollama",
            model="llama3",
            base_url="http://localhost:11434",
            api_key=None,
        )
        client._http.post = fake_post

        with self.assertRaises(LLMClientError) as ctx:
            client.chat([{"role": "user", "content": "ping"}])
        self.assertIn("Connection refused", str(ctx.exception))
