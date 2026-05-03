# tests/test_agents_init.py
import pytest

from agents import build_llm_clients, safe_parse_json


def test_safe_parse_json_plain():
    assert safe_parse_json('{"a": 1}') == {"a": 1}


def test_safe_parse_json_markdown_fenced():
    payload = '```json\n{"a": 1, "b": "x"}\n```'
    assert safe_parse_json(payload) == {"a": 1, "b": "x"}


def test_safe_parse_json_bare_fence():
    payload = '```\n{"a": 1}\n```'
    assert safe_parse_json(payload) == {"a": 1}


def test_build_llm_clients_returns_two_models():
    clients = build_llm_clients("sk-ant-fake-test-key")
    assert clients.reasoning is not None
    assert clients.fast is not None
    # Models bind at call time; just verify the configured names are correct.
    assert "sonnet" in clients.reasoning.model.lower()
    assert "haiku" in clients.fast.model.lower()


def test_build_llm_clients_rejects_empty_key():
    with pytest.raises(ValueError, match="anthropic"):
        build_llm_clients("")


def test_degraded_signal_shape():
    from agents import degraded_signal
    out = degraded_signal("price", "Technical Analysis", "No data for ZZZZ")
    assert "agent_signals" in out
    sig = out["agent_signals"][0]
    assert sig["agent"] == "price"
    assert sig["signal"] == "NEUTRAL"
    assert sig["confidence"] == 0.0
    assert sig["degraded"] is True
    assert sig["error"] is None
    assert sig["section_markdown"].startswith("## Technical Analysis")
    assert "No data for ZZZZ" in sig["section_markdown"]
    assert sig["raw_data"] == {}


def test_degraded_signal_with_raw_and_error():
    from agents import degraded_signal
    out = degraded_signal("risk", "Risk Profile", "boom", raw={"foo": 1}, error="trace")
    sig = out["agent_signals"][0]
    assert sig["raw_data"] == {"foo": 1}
    assert sig["error"] == "trace"


def test_tool_def_to_anthropic_shape():
    from agents import ToolDef
    t = ToolDef(
        name="echo",
        description="Echo input",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=lambda args: {"echoed": args.get("x")},
    )
    spec = t.to_anthropic()
    assert spec["name"] == "echo"
    assert spec["description"] == "Echo input"
    assert spec["input_schema"]["type"] == "object"
    assert "handler" not in spec  # handler is local-only


def test_run_with_tools_returns_parsed_json_no_tool_use(monkeypatch):
    """Happy path: model emits final JSON immediately, no tool calls."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    fake_block = MagicMock()
    fake_block.text = '{"signal": "BULLISH", "confidence": 0.7}'
    fake_resp = MagicMock(stop_reason="end_turn", content=[fake_block])

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp

    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="You are a test agent.",
        user_prompt="Analyze MSFT.",
        tools=[],
    )
    assert out == {"signal": "BULLISH", "confidence": 0.7}


def test_run_with_tools_executes_tool_then_returns_json(monkeypatch):
    """Model emits a tool_use, we run handler, model emits final JSON."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    # Iteration 0: model asks for tool 'foo' with input {"a": 1}
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_01"
    tool_use_block.name = "foo"
    tool_use_block.input = {"a": 1}
    resp_iter0 = MagicMock(stop_reason="tool_use", content=[tool_use_block])

    # Iteration 1: model returns final text
    text_block = MagicMock()
    text_block.text = '{"signal": "NEUTRAL", "confidence": 0.5}'
    resp_iter1 = MagicMock(stop_reason="end_turn", content=[text_block])

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [resp_iter0, resp_iter1]

    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    handler_mock = MagicMock(return_value={"result": 42})
    tool = agents_mod.ToolDef(
        name="foo",
        description="test tool",
        input_schema={"type": "object"},
        handler=handler_mock,
    )

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="sys",
        user_prompt="user",
        tools=[tool],
    )
    handler_mock.assert_called_once_with({"a": 1})
    assert out == {"signal": "NEUTRAL", "confidence": 0.5}


def test_run_with_tools_caps_at_max_iterations(monkeypatch):
    """After max_iterations tool-use turns, the call must omit tools."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_X"
    tool_use_block.name = "foo"
    tool_use_block.input = {}

    resp_tool = MagicMock(stop_reason="tool_use", content=[tool_use_block])
    final_text = MagicMock()
    final_text.text = '{"signal": "HOLD", "confidence": 0.3}'
    resp_final = MagicMock(stop_reason="end_turn", content=[final_text])

    # 3 tool-use turns + 1 forced final = 4 calls
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [resp_tool, resp_tool, resp_tool, resp_final]

    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    tool = agents_mod.ToolDef(
        name="foo", description="x", input_schema={"type": "object"},
        handler=lambda args: {"ok": True},
    )

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="sys", user_prompt="user", tools=[tool],
        max_iterations=3,
    )
    # 4th (final) call must omit `tools` so Claude is forced to produce text.
    final_call_kwargs = fake_client.messages.create.call_args_list[-1].kwargs
    assert "tools" not in final_call_kwargs
    assert out == {"signal": "HOLD", "confidence": 0.3}


def test_run_with_tools_handler_exception_returned_as_tool_result(monkeypatch):
    """Handler raises -> tool_result content is the error string, loop continues."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_e"
    tool_use_block.name = "broken"
    tool_use_block.input = {}
    resp_tool = MagicMock(stop_reason="tool_use", content=[tool_use_block])

    final_text = MagicMock()
    final_text.text = '{"signal": "NEUTRAL", "confidence": 0.4}'
    resp_final = MagicMock(stop_reason="end_turn", content=[final_text])

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [resp_tool, resp_final]
    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    def boom(_args):
        raise RuntimeError("bang")

    tool = agents_mod.ToolDef(
        name="broken", description="x", input_schema={"type": "object"},
        handler=boom,
    )

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="s", user_prompt="u", tools=[tool],
    )
    assert out == {"signal": "NEUTRAL", "confidence": 0.4}
    # Verify the second call sent a tool_result with an "error" key.
    second_call_messages = fake_client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert "bang" in tool_result_msg["content"][0]["content"]
