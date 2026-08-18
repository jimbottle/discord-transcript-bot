"""LLM provider chain backing the `/ask` command.

OpenRouter (primary) -> Cerebras paid (fallback) -> Ollama (local last
resort). See :mod:`src.llm.chain` for the failover semantics.
"""
