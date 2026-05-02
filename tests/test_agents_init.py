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
