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
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

from anthropic import Anthropic
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


# ---------------------------------------------------------------------------
# v2.1: Tool-use loop helpers
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS = 3


@dataclass
class ToolDef:
    """A single on-demand tool available to a specialist agent."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], object]   # local executor; returns JSON-serializable

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _content_blocks_text(blocks) -> str:
    """Concatenate text from a list of TextBlock / dicts."""
    out = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text:
            out.append(text)
    return "".join(out)


def _format_tool_result(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception as exc:
        return json.dumps({"error": f"unserializable: {exc!s}"})


def run_with_tools(
    api_key: str,
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[ToolDef],
    model: str = REASONING_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = 1500,
    temperature: float = 0.1,
) -> dict:
    """Run a Claude tool-use loop with hard iteration cap and prompt caching.

    The system prompt block is sent with cache_control on every turn so that
    Anthropic can serve subsequent turns from cache (5-minute TTL).

    Returns the parsed JSON dict from the final text response.

    Raises ValueError if the model never produces a parseable JSON text block
    after `max_iterations + 1` attempts.
    """
    if not api_key:
        raise ValueError("api_key is required")

    client = Anthropic(api_key=api_key)
    tool_specs = [t.to_anthropic() for t in tools]
    handlers = {t.name: t.handler for t in tools}

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    for iteration in range(max_iterations + 1):
        # Force final text on the LAST iteration (no more tool calls allowed).
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=list(messages),
        )
        if iteration < max_iterations and tool_specs:
            kwargs["tools"] = tool_specs
        # Final iteration: omit `tools` entirely (instead of tool_choice="none")
        # so the model has no schema to call against and must emit text. This
        # also yields a smaller cache key for the forced-text turn.

        resp = client.messages.create(**kwargs)

        # Append assistant turn to message history.
        messages.append({"role": "assistant", "content": resp.content})

        # If stop_reason is tool_use, execute every tool_use block in this turn
        # and feed all results back as a single user message.
        if resp.stop_reason == "tool_use" and iteration < max_iterations:
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = block.name
                args = block.input or {}
                try:
                    handler = handlers[name]
                    result = handler(args)
                    result_text = _format_tool_result(result)
                except KeyError:
                    result_text = json.dumps({"error": f"unknown tool {name}"})
                except Exception as exc:
                    result_text = json.dumps({"error": str(exc)[:200]})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Otherwise, parse the final text and return.
        text = _content_blocks_text(resp.content)
        if not text:
            continue
        try:
            return safe_parse_json(text)
        except Exception:
            # If parse fails on this turn, loop will retry up to max_iterations.
            continue

    raise ValueError(
        f"run_with_tools: model did not produce parseable JSON after "
        f"{max_iterations + 1} iterations"
    )
