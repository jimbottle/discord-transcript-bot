"""Unit tests for the shared /ask model resolver.

Guards the env plumbing roborev flagged on commit 2d16d19: the default
was duplicated across main.py and health.py with no test, and an empty
ASK_OLLAMA_MODEL= silently produced model="".
"""

from src.config.ollama_config import (
    DEFAULT_ASK_MODEL,
    get_ask_model,
    is_ollama_pullable,
)


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


def test_default_is_the_benchmarked_model():
    # Was "ai/mistral:latest" — a Docker Hub name Ollama's registry 404s, so
    # /ask could never work unconfigured. Now the local-models bake-off winner.
    assert DEFAULT_ASK_MODEL == "gemma4:26b"


def test_default_is_actually_pullable_from_ollama():
    """The bug class this whole module exists to prevent: a default nobody can
    install. Any future change to DEFAULT_ASK_MODEL trips this."""
    assert is_ollama_pullable(DEFAULT_ASK_MODEL)


def test_docker_hub_names_are_rejected():
    assert not is_ollama_pullable("ai/mistral:latest")
    assert not is_ollama_pullable("AI/Mistral:Latest")  # case-insensitive
    assert not is_ollama_pullable("  ai/smollm2  ")  # tolerates stray spacing


def test_ollama_namespaced_names_are_accepted():
    """Deliberately narrow: Ollama DOES serve user- and hf-namespaced models,
    so a blanket 'contains a slash' rule would reject valid choices."""
    for name in ("gemma4:26b", "mistral:latest", "someuser/model", "hf.co/org/m"):
        assert is_ollama_pullable(name), name
