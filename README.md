# simple_agent_loop

A minimal agent loop for tool-using language models. ~250 lines. Handles
message format conversion, parallel tool execution, session compaction,
and subagent composition.

## Install

```
pip install simple-agent-loop
```

## Setup

```python
import anthropic
import json
import simple_agent_loop as sal

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

def invoke_model(tools, session):
    kwargs = dict(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        messages=session["messages"],
    )
    if "system" in session:
        kwargs["system"] = session["system"]
    if tools:
        kwargs["tools"] = tools
    return client.messages.create(**kwargs).to_dict()
```

## Hello World

No tools, single turn -- the model just responds:

```python
session = init_session(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say hello in three languages.",
)
result = agent_loop(invoke_model, [], session, max_iterations=1)
print(response(result)["content"])
```

## Tool-Using Agent

Define tools as Anthropic tool schemas and provide handler functions. The
handler receives tool input as keyword arguments and returns a string.

```python
import requests

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }
]

def get_weather(city):
    resp = requests.get(f"https://wttr.in/{city}?format=j1")
    data = resp.json()["current_condition"][0]
    return json.dumps({
        "city": city,
        "temp_c": data["temp_C"],
        "description": data["weatherDesc"][0]["value"],
    })

session = init_session(
    system_prompt="You answer weather questions. Use the get_weather tool.",
    user_prompt="What's the weather in Tokyo and Paris?",
)
result = agent_loop(
    invoke_model, tools, session,
    tool_handlers={"get_weather": get_weather},
)
print(response(result)["content"])
```

The model will call get_weather twice (in parallel), see the results, and
respond with a summary. The loop runs until the model responds without
making any tool calls.

## Subagents

A subagent is a tool handler that runs its own agent loop. The outer agent
calls it like any tool and gets back a string result.

### Example: Text Compressor (compressor.py)

A coordinator agent iteratively compresses text using two subagents:
a shortener and a quality judge.

```python
# Subagent: compresses text
def shorten(text):
    session = init_session(
        system_prompt="Rewrite the text to half its length. Output ONLY the result.",
        user_prompt=text,
    )
    result = agent_loop(invoke_model, [], session, name="shortener", max_iterations=1)
    shortened = response(result)["content"]
    ratio = len(shortened) / len(text)
    return json.dumps({"compression_ratio": round(ratio, 3), "shortened_text": shortened})

# Subagent: judges compression quality
def judge(original, shortened):
    session = init_session(
        system_prompt=(
            "Compare original and shortened text. Return ONLY JSON: "
            '{"verdict": "acceptable", "reason": "..."} or '
            '{"verdict": "too_lossy", "reason": "..."}'
        ),
        user_prompt=f"ORIGINAL:\n{original}\n\nSHORTENED:\n{shortened}",
    )
    result = agent_loop(invoke_model, [], session, name="judge", max_iterations=1)
    return response(result)["content"]
```

The coordinator has tools for `shorten` and `judge`, and its system prompt
tells it to loop: shorten, judge, stop if too_lossy or diminishing returns,
otherwise shorten again. Each subagent is a one-shot agent loop
(max_iterations=1) with no tools of its own.

### Example: Transform Rule Derivation (derive_transform.py)

A more complex example with four subagents and a coordinator. Given a
source text and target text, it derives general transformation rules and
specific info that together reproduce the target from the source.

```python
# Subagent: applies rules + specific info to source text
def edit(text, rules, specific_info):
    session = init_session(
        system_prompt="Apply the rules to the text using the specific info. Output ONLY the result.",
        user_prompt=f"SOURCE TEXT:\n{text}\n\nRULES:\n{rules}\n\nSPECIFIC INFO:\n{specific_info}",
    )
    result = agent_loop(invoke_model, [], session, name="editor", max_iterations=1)
    return response(result)["content"]

# Subagent: scores how close the output is to the target
def judge_similarity(editor_output, target):
    session = init_session(
        system_prompt='Compare texts. Return JSON: {"score": 0-100, "differences": "..."}',
        user_prompt=f"EDITOR OUTPUT:\n{editor_output}\n\nTARGET:\n{target}",
    )
    result = agent_loop(invoke_model, [], session, name="similarity-judge", max_iterations=1)
    return response(result)["content"]

# Subagent: checks rules are abstract (no specific content leaked in)
def judge_generality(rules):
    ...

# Subagent: checks specific_info is a flat fact list
def judge_specific_info(specific_info):
    ...
```

The coordinator calls `edit`, then calls all three judges in parallel,
refines based on scores, and repeats until all judges score above 90.
Tool calls within a single model response execute in parallel automatically.

## API Reference

### Session Management

- `init_session(system_prompt, user_prompt)` - Create a new session
- `extend_session(session, message)` - Append a message to the session
- `send(session, user_message)` - Add a user message to the session
- `fork_session(session)` - Deep copy a session for branching
- `response(session)` - Get the last assistant message, or None

### Agent Loop

- `agent_loop(invoke_model, tools, session, tool_handlers=None, name=None, max_iterations=None)`
  - `invoke_model(tools, session)` - Function that calls the model API
  - `tools` - List of Anthropic tool schemas ([] for no tools)
  - `session` - Session dict from init_session
  - `tool_handlers` - Dict mapping tool names to handler functions
  - `name` - Agent name for log output
  - `max_iterations` - Max model calls before stopping (None = unlimited)
  - Returns the session with all messages appended

### Message Format

Messages use a generic format independent of any API:

    {"role": "system", "content": "..."}
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}
    {"type": "thinking", "content": "..."}
    {"type": "tool_call", "name": "...", "id": "...", "input": {...}}
    {"type": "tool_result", "id": "...", "output": "..."}

Conversion to/from Anthropic API format is handled by `to_api_messages()`
and `parse_response()`.
