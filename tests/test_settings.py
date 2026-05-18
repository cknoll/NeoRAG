"""Tests for NeoRAGSettings and load_settings() (improvement-plan2 step 7)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import pytest

from neorag.config import (
    DEFAULT_COLLECTION,
    LLM_MODEL,
    LLM_PROVIDER,
    TOP_K_BASE,
    TOP_K_FINAL,
    EvalSettings,
    LLMSettings,
    NeoRAGSettings,
    load_settings,
)


class TestNeoRAGSettingsDefaults(unittest.TestCase):
    def test_llm_defaults_match_module_constants(self):
        s = NeoRAGSettings()
        self.assertEqual(s.llm.provider, LLM_PROVIDER)
        self.assertEqual(s.llm.model, LLM_MODEL)

    def test_retrieval_defaults_match_module_constants(self):
        s = NeoRAGSettings()
        self.assertEqual(s.top_k_base, TOP_K_BASE)
        self.assertEqual(s.top_k_final, TOP_K_FINAL)
        self.assertEqual(s.default_collection, DEFAULT_COLLECTION)

    def test_eval_defaults(self):
        s = NeoRAGSettings()
        self.assertIsNone(s.eval.test_path)
        self.assertEqual(s.eval.max_iter, 3)
        self.assertEqual(s.eval.feedback_granularity, "per_violation")
        self.assertFalse(s.eval.no_refine)

    def test_instantiated_without_arguments(self):
        s = NeoRAGSettings()
        self.assertIsInstance(s.llm, LLMSettings)
        self.assertIsInstance(s.eval, EvalSettings)

    def test_init_kwargs_override_defaults(self):
        s = NeoRAGSettings(llm=LLMSettings(provider="stub", model="stub"))
        self.assertEqual(s.llm.provider, "stub")
        # Other fields unaffected
        self.assertEqual(s.top_k_base, TOP_K_BASE)


class TestLoadSettingsFromToml(unittest.TestCase):
    def _write_toml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "test_settings.toml"
        p.write_text(content)
        return p

    def test_no_path_returns_default_settings(self, tmp_path=None):
        s = load_settings(None)
        self.assertIsInstance(s, NeoRAGSettings)
        self.assertEqual(s.llm.provider, LLM_PROVIDER)

    def test_toml_llm_section_loaded(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        toml = self._write_toml(
            tmp_path,
            '[llm]\nprovider = "stub"\nmodel = "my-stub-model"\n',
        )
        s = load_settings(toml)
        self.assertEqual(s.llm.provider, "stub")
        self.assertEqual(s.llm.model, "my-stub-model")

    def test_toml_eval_section_loaded(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        toml = self._write_toml(
            tmp_path,
            '[eval]\ntest_path = "tests/fixtures/demo_test.json"\nmax_iter = 5\n',
        )
        s = load_settings(toml)
        self.assertEqual(s.eval.test_path, "tests/fixtures/demo_test.json")
        self.assertEqual(s.eval.max_iter, 5)

    def test_toml_top_level_keys_loaded(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        toml = self._write_toml(
            tmp_path,
            "top_k_final = 7\ndefault_collection = \"my_corpus\"\n",
        )
        s = load_settings(toml)
        self.assertEqual(s.top_k_final, 7)
        self.assertEqual(s.default_collection, "my_corpus")

    def test_unspecified_toml_fields_use_defaults(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        toml = self._write_toml(tmp_path, '[llm]\nprovider = "stub"\n')
        s = load_settings(toml)
        # max_tokens not set → default
        self.assertEqual(s.llm.max_tokens, LLMSettings().max_tokens)

    def test_demo_toml_loads_without_error(self):
        demo = Path(__file__).parent.parent / "configs" / "demo.toml"
        self.assertTrue(demo.is_file(), "configs/demo.toml should exist")
        s = load_settings(demo)
        self.assertEqual(s.llm.provider, "stub")
        self.assertIsNotNone(s.eval.test_path)

    def test_env_var_overrides_toml(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        toml = self._write_toml(tmp_path, '[llm]\nprovider = "stub"\n')
        old = os.environ.get("NEORAG_LLM__PROVIDER")
        try:
            os.environ["NEORAG_LLM__PROVIDER"] = "openai"
            s = load_settings(toml)
            self.assertEqual(s.llm.provider, "openai")
        finally:
            if old is None:
                os.environ.pop("NEORAG_LLM__PROVIDER", None)
            else:
                os.environ["NEORAG_LLM__PROVIDER"] = old


class TestBackwardsCompatibility(unittest.TestCase):
    """Verify that the existing module-level constants are still importable."""

    def test_module_constants_still_exist(self):
        from neorag import config

        for name in (
            "LLM_PROVIDER",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LLM_MAX_TOKENS",
            "LLM_TIMEOUT_S",
            "TOP_K_BASE",
            "TOP_K_FINAL",
            "DEFAULT_COLLECTION",
            "CORPUS_DEFAULTS",
            "EMBEDDING_MODEL",
            "RERANK_MODEL",
            "QDRANT_PATH",
        ):
            self.assertTrue(hasattr(config, name), f"config.{name} missing")

    def test_validate_dirs_still_callable(self):
        from neorag.config import validate_dirs

        self.assertTrue(callable(validate_dirs))

    def test_load_api_key_from_toml_still_callable(self):
        from neorag.config import load_api_key_from_toml

        self.assertTrue(callable(load_api_key_from_toml))
