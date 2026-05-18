"""Unit tests for the shared /ask model resolver.

Guards the env plumbing roborev flagged on commit 2d16d19: the default
was duplicated across main.py and health.py with no test, and an empty
ASK_OLLAMA_MODEL= silently produced model="".
"""
from src.config.ollama_config import DEFAULT_ASK_MODEL, get_ask_model


def test_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("ASK_OLLAMA_MODEL", raising=False)
    assert get_ask_model() == DEFAULT_ASK_MODEL


def test_env_override_used(monkeypatch):
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "gemma4:26b")
    assert get_ask_model() == "gemma4:26b"


def test_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "")
    assert get_ask_model() == DEFAULT_ASK_MODEL


def test_whitespace_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "   \t ")
    assert get_ask_model() == DEFAULT_ASK_MODEL


def test_surrounding_whitespace_trimmed(monkeypatch):
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "  llama3.2:3b  ")
    assert get_ask_model() == "llama3.2:3b"


def test_default_matches_prior_behavior():
    # The historical hardcoded value, preserved when the var is unset.
    assert DEFAULT_ASK_MODEL == "ai/mistral:latest"
