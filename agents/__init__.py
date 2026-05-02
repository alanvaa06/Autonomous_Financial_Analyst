# agents/__init__.py
"""Shared LLM clients + JSON helper for MarketMind v2 specialist agents.

`build_llm_clients(anthropic_key)` is the only entry point. It returns a
NamedTuple of two ChatAnthropic clients (Sonnet + Haiku) bound to the
session-scoped key. No module-level client is ever constructed from
`os.environ` on the public BYO-key path.

A process-wide asyncio semaphore caps in-flight Anthropic requests at 3 per
key to avoid 429s during fan-out. (LangGraph's superstep will start all 5
specialist nodes concurrently; the semaphore queues two of them briefly.)
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import NamedTuple

from langchain_anthropic import ChatAnthropic

# Per-key concurrency semaphores. Keyed by the API key string; ephemeral,
# session-scoped, GC'd when the LLMClients reference is dropped.
_semaphores: dict[str, asyncio.Semaphore] = {}
_semaphores_lock = threading.Lock()

_MAX_CONCURRENT_PER_KEY = 3

REASONING_MODEL = "claude-sonnet-4-6"
FAST_MODEL = "claude-haiku-4-5-20251001"


class LLMClients(NamedTuple):
    reasoning: ChatAnthropic
    fast: ChatAnthropic
    semaphore_key: str


def build_llm_clients(anthropic_key: str) -> LLMClients:
    if not anthropic_key:
        raise ValueError("anthropic_key is required (BYO-key)")

    reasoning = ChatAnthropic(
        model=REASONING_MODEL,
        api_key=anthropic_key,
        temperature=0.1,
        max_tokens=1500,
    )
    fast = ChatAnthropic(
        model=FAST_MODEL,
        api_key=anthropic_key,
        temperature=0.1,
        max_tokens=800,
    )
    return LLMClients(reasoning=reasoning, fast=fast, semaphore_key=anthropic_key)


def get_semaphore(key: str) -> asyncio.Semaphore:
    """Return the per-key semaphore, creating it lazily."""
    with _semaphores_lock:
        sem = _semaphores.get(key)
        if sem is None:
            sem = asyncio.Semaphore(_MAX_CONCURRENT_PER_KEY)
            _semaphores[key] = sem
        return sem


def safe_parse_json(content: str) -> dict:
    """Parse LLM JSON output. Strips ```json ... ``` and ``` ... ``` fences.

    Llama and Claude both occasionally wrap JSON in markdown fences even when
    instructed otherwise. This is a defensive parser shared by every agent.
    """
    if not isinstance(content, str):
        raise TypeError(f"safe_parse_json expects str, got {type(content)!r}")

    text = content.strip()
    if text.startswith("```"):
        # Remove the opening fence (with optional language tag) and the closing fence.
        first_newline = text.find("\n")
        if first_newline == -1:
            raise json.JSONDecodeError("Empty fenced block", text, 0)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
        text = text.strip()
    return json.loads(text)
