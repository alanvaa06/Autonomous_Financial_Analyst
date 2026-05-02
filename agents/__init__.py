# agents/__init__.py
"""Shared LLM clients + JSON helper + degraded-signal helper for MarketMind v2.

`build_llm_clients(anthropic_key)` is the only entry point. It returns a
NamedTuple of two ChatAnthropic clients (Sonnet + Haiku) bound to the
session-scoped key. No module-level client is ever constructed from
`os.environ` on the public BYO-key path.

`degraded_signal(...)` is the shared factory for the AgentSignal returned by
every specialist when it cannot produce real output (missing key, no data,
LLM error, etc.). One implementation, used by all five specialists.
"""

from __future__ import annotations

import json
from typing import NamedTuple, Optional

from langchain_anthropic import ChatAnthropic

from state import AgentSignal

REASONING_MODEL = "claude-sonnet-4-6"
FAST_MODEL = "claude-haiku-4-5-20251001"


class LLMClients(NamedTuple):
    reasoning: ChatAnthropic
    fast: ChatAnthropic


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
    return LLMClients(reasoning=reasoning, fast=fast)


def safe_parse_json(content: str) -> dict:
    """Parse LLM JSON output. Strips ```json ... ``` and ``` ... ``` fences.

    Llama and Claude both occasionally wrap JSON in markdown fences even when
    instructed otherwise. This is a defensive parser shared by every agent.
    """
    if not isinstance(content, str):
        raise TypeError(f"safe_parse_json expects str, got {type(content)!r}")

    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline == -1:
            raise json.JSONDecodeError("Empty fenced block", text, 0)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
        text = text.strip()
    return json.loads(text)


def degraded_signal(
    agent: str,
    section_title: str,
    reason: str,
    raw: Optional[dict] = None,
    error: Optional[str] = None,
) -> dict:
    """Construct the standard degraded AgentSignal envelope.

    Returns the LangGraph node-update shape: `{"agent_signals": [AgentSignal]}`.
    Every specialist uses this when it cannot produce a real signal — missing
    key, no data, LLM error, or any other graceful-degradation case.
    """
    return {"agent_signals": [AgentSignal(
        agent=agent,
        signal="NEUTRAL",
        confidence=0.0,
        summary=reason,
        section_markdown=f"## {section_title}\n_Unavailable: {reason}_",
        raw_data=raw or {},
        degraded=True,
        error=error,
    )]}
