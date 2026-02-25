from simple_agent_loop import *
import json
import pytest


TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers together. Use this for any addition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    }
]


def make_invoke_model():
    """Create a mock invoke_model that expects: user msg -> thinking + tool_call -> tool_result -> final answer."""
    call_count = 0

    def invoke_model(tools, session):
        nonlocal call_count
        call_count += 1
        messages = session["messages"]

        if call_count == 1:
            if not any(m.get("role") == "user" for m in messages):
                raise ValueError("Expected a user message on first call")
            return [
                {"type": "thinking", "content": "I need to add these two numbers.", "signature": "sig_abc123"},
                {"type": "tool_call", "name": "add", "id": "call_001", "input": {"a": 98765432101234, "b": 12345678909876}},
            ]
        elif call_count == 2:
            if not any(m.get("type") == "tool_result" for m in messages):
                raise ValueError("Expected a tool_result on second call")
            tool_result = next(m for m in messages if m.get("type") == "tool_result")
            result_value = json.loads(tool_result["output"])["result"]
            return [
                {"role": "assistant", "content": f"The answer is {result_value}.", "ts": now()},
            ]
        else:
            raise ValueError(f"Unexpected call #{call_count}")

    return invoke_model


def add(a, b):
    return json.dumps({"result": a + b})


class TestAgentLoop:
    def test_tool_call_and_response(self):
        """Full loop: user question -> thinking + tool call -> tool result -> final answer."""
        session = {
            "messages": [
                {"role": "user", "content": "What is 98765432101234 + 12345678909876?"}
            ]
        }
        result = agent_loop(make_invoke_model(), TOOLS, session, tool_handlers={"add": add})
        final = response(result)
        assert final is not None
        assert "111111111011110" in final["content"]

    def test_session_contains_all_message_types(self):
        """Verify the session accumulates thinking, tool_call, tool_result, and assistant messages."""
        session = {
            "messages": [
                {"role": "user", "content": "What is 98765432101234 + 12345678909876?"}
            ]
        }
        result = agent_loop(make_invoke_model(), TOOLS, session, tool_handlers={"add": add})
        msgs = result["messages"]
        types = [(m.get("role"), m.get("type")) for m in msgs]
        assert ("user", None) in types
        assert (None, "thinking") in types
        assert (None, "tool_call") in types
        assert (None, "tool_result") in types
        assert ("assistant", None) in types

    def test_tool_result_is_correct(self):
        """Verify the tool handler is called and produces the right output."""
        session = {
            "messages": [
                {"role": "user", "content": "What is 98765432101234 + 12345678909876?"}
            ]
        }
        result = agent_loop(make_invoke_model(), TOOLS, session, tool_handlers={"add": add})
        tool_results = [m for m in result["messages"] if m.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert json.loads(tool_results[0]["output"])["result"] == 111111111011110

    def test_no_user_message_errors(self):
        """invoke_model should error if there's no user message."""
        session = {"messages": []}
        with pytest.raises(ValueError, match="Expected a user message"):
            agent_loop(make_invoke_model(), TOOLS, session, tool_handlers={"add": add})

    def test_stops_without_tool_calls(self):
        """Loop should stop after a single call if the model returns no tool calls."""
        def invoke_once(tools, session):
            return [
                {"role": "assistant", "content": "No tools needed.", "ts": now()},
            ]

        session = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        result = agent_loop(invoke_once, TOOLS, session, tool_handlers={"add": add})
        final = response(result)
        assert final["content"] == "No tools needed."
        # Should only have user + assistant (no tool calls/results)
        assert not any(m.get("type") == "tool_call" for m in result["messages"])

    def test_sequential_mode(self):
        """parallel=False should produce the same results as parallel mode."""
        session = {
            "messages": [
                {"role": "user", "content": "What is 98765432101234 + 12345678909876?"}
            ]
        }
        result = agent_loop(make_invoke_model(), TOOLS, session, tool_handlers={"add": add}, parallel=False)
        final = response(result)
        assert final is not None
        assert "111111111011110" in final["content"]
        tool_results = [m for m in result["messages"] if m.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert json.loads(tool_results[0]["output"])["result"] == 111111111011110

    def test_max_iterations(self):
        """Loop should respect max_iterations even if model keeps returning tool calls."""
        def invoke_always_tool(tools, session):
            return [
                {"type": "tool_call", "name": "add", "id": "call_loop", "input": {"a": 1, "b": 2}},
            ]

        session = {
            "messages": [
                {"role": "user", "content": "Keep adding"}
            ]
        }
        result = agent_loop(invoke_always_tool, TOOLS, session, tool_handlers={"add": add}, max_iterations=3)
        tool_calls = [m for m in result["messages"] if m.get("type") == "tool_call"]
        assert len(tool_calls) == 3
