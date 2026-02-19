import anthropic
from simple_agent_loop import *
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

def invoke_model(tools, session):
    # Convert generic messages to Anthropic API format
    system_prompt = None
    api_messages = []
    assistant_blocks = []
    tool_result_blocks = []

    def flush_assistant():
        nonlocal assistant_blocks
        if assistant_blocks:
            api_messages.append({"role": "assistant", "content": assistant_blocks})
            assistant_blocks = []

    def flush_tool_results():
        nonlocal tool_result_blocks
        if tool_result_blocks:
            api_messages.append({"role": "user", "content": tool_result_blocks})
            tool_result_blocks = []

    for msg in session["messages"]:
        role = msg.get("role")
        msg_type = msg.get("type")
        if role == "system":
            system_prompt = msg["content"]
        elif role == "user":
            flush_assistant()
            flush_tool_results()
            api_messages.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            flush_tool_results()
            assistant_blocks.append({"type": "text", "text": msg["content"]})
        elif msg_type == "thinking":
            flush_tool_results()
            block = {"type": "thinking", "thinking": msg["content"]}
            if "signature" in msg:
                block["signature"] = msg["signature"]
            assistant_blocks.append(block)
        elif msg_type == "tool_call":
            flush_tool_results()
            assistant_blocks.append({
                "type": "tool_use", "id": msg["id"],
                "name": msg["name"], "input": msg["input"],
            })
        elif msg_type == "tool_result":
            flush_assistant()
            output = msg["output"]
            tool_result_blocks.append({
                "type": "tool_result", "tool_use_id": msg["id"],
                "content": output if isinstance(output, str) else json.dumps(output),
            })
    flush_assistant()
    flush_tool_results()

    # Call the model
    kwargs = dict(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        messages=api_messages,
        thinking={"type": "enabled", "budget_tokens": 10000},
        tools=tools,
    )
    if system_prompt:
        kwargs["system"] = system_prompt
    api_response = client.messages.create(**kwargs).to_dict()

    # Parse response back to generic messages
    messages = []
    for block in api_response.get("content", []):
        if block["type"] == "thinking":
            msg = {"type": "thinking", "content": block["thinking"], "ts": now()}
            if "signature" in block:
                msg["signature"] = block["signature"]
            messages.append(msg)
        elif block["type"] == "text" and block["text"]:
            messages.append({"role": "assistant", "content": block["text"], "ts": now()})
        elif block["type"] == "tool_use":
            messages.append({
                "type": "tool_call", "name": block["name"],
                "id": block["id"], "input": block["input"],
            })
    return messages


tools = [
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

def add(a, b):
    return json.dumps({"result": a + b})

session = {
    "messages": [
        {"role": "user", "content": "What is 98765432101234 + 12345678909876?"}
    ]
}

result = agent_loop(invoke_model, tools, session, tool_handlers={"add": add})

print("\n" + "-"*80)
print(response(result)["content"])
