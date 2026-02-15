from datetime import datetime, timezone
import copy
import json
import concurrent.futures


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_session(system_prompt, user_prompt):
    return {
        "messages": [
            {"role": "system", "content": system_prompt, "ts": now()},
            {"role": "user", "content": user_prompt, "ts": now()},
        ]
    }


def extend_session(session, message):
    session["messages"].append(message)


def send(session, user_message):
    extend_session(session, {"role": "user", "content": user_message, "ts": now()})


def fork_session(session):
    return copy.deepcopy(session)


def response(session):
    for msg in reversed(session["messages"]):
        if msg.get("role") == "assistant":
            return msg
    return None


def to_api_messages(messages):
    """Convert generic messages to Anthropic API format.

    Groups consecutive thinking/assistant/tool_call into a single assistant message,
    and consecutive tool_results into a single user message.
    """
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

    for msg in messages:
        role = msg.get("role")
        msg_type = msg.get("type")

        if role == "system":
            continue

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
                "type": "tool_use",
                "id": msg["id"],
                "name": msg["name"],
                "input": msg["input"],
            })

        elif msg_type == "tool_result":
            flush_assistant()
            output = msg["output"]
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": msg["id"],
                "content": output if isinstance(output, str) else json.dumps(output),
            })

    flush_assistant()
    flush_tool_results()
    return api_messages


def parse_response(api_response):
    """Parse Anthropic API response dict into generic messages."""
    messages = []
    ts = now()

    for block in api_response.get("content", []):
        block_type = block["type"]

        if block_type == "thinking":
            msg = {"type": "thinking", "content": block["thinking"], "ts": ts}
            if "signature" in block:
                msg["signature"] = block["signature"]
            messages.append(msg)

        elif block_type == "text":
            if block["text"]:
                messages.append({"role": "assistant", "content": block["text"], "ts": ts})

        elif block_type == "tool_use":
            messages.append({
                "type": "tool_call",
                "name": block["name"],
                "id": block["id"],
                "input": block["input"],
            })

    return messages


def execute_tool_calls(tool_calls, tool_handlers):
    """Execute tool calls in parallel via ThreadPoolExecutor. Errors are caught and returned as results."""
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tc = {}
        for tc in tool_calls:
            handler = tool_handlers.get(tc["name"])
            if handler is None:
                results.append({
                    "type": "tool_result",
                    "id": tc["id"],
                    "output": f"Error: unknown tool '{tc['name']}'",
                })
                continue
            future = executor.submit(handler, **tc["input"])
            future_to_tc[future] = tc

        for future in concurrent.futures.as_completed(future_to_tc):
            tc = future_to_tc[future]
            try:
                output = future.result()
                results.append({
                    "type": "tool_result",
                    "id": tc["id"],
                    "output": output,
                })
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "id": tc["id"],
                    "output": f"Error: {e}",
                })

    return results


def log(message, name=None):
    ts = message.get("ts", now())
    agent = name or "agent"

    role = message.get("role", "")
    msg_type = message.get("type", "")
    label = role or msg_type

    if msg_type == "thinking":
        content = message.get("content", "")
    elif msg_type == "tool_call":
        content = f"{message['name']}({json.dumps(message['input'])})"
    elif msg_type == "tool_result":
        output = message.get("output", "")
        content = output if isinstance(output, str) else json.dumps(output)
    else:
        content = message.get("content", "")

    content = content.replace("\n", " ").strip()
    line = f"{ts} {agent} {label} {content}"
    if len(line) > 120:
        line = line[:117] + "..."
    print(line)


def agent_loop(invoke_model, tools, session, tool_handlers=None, name=None, max_iterations=None):
    if tool_handlers is None:
        tool_handlers = {}

    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1

        system_prompts = [m["content"] for m in session["messages"] if m.get("role") == "system"]
        api_session = {"messages": to_api_messages(session["messages"])}
        if system_prompts:
            api_session["system"] = system_prompts[0]
        api_response = invoke_model(tools, api_session)

        messages = parse_response(api_response)
        for msg in messages:
            extend_session(session, msg)
            log(msg, name)

        tool_calls = [m for m in messages if m.get("type") == "tool_call"]
        if not tool_calls:
            break

        results = execute_tool_calls(tool_calls, tool_handlers)
        for result in results:
            extend_session(session, result)
            log(result, name)

    return session
