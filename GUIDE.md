# How to Implement an Agent Loop

An agent loop lets a language model use tools repeatedly until it finishes a task. The model decides what to do, calls tools, sees results, and keeps going until it has an answer. This guide covers the core concepts and data structures needed to build one from scratch in any language.

## Core Data Structure: The Session

A session is a list of messages. Each message has a role or type and some content:

```
{ role: "system",    content: "..." }       -- system prompt
{ role: "user",      content: "..." }       -- user input
{ role: "assistant", content: "..." }       -- model text output
{ type: "thinking",  content: "..." }       -- model reasoning (if supported)
{ type: "tool_call", name: "...", id: "...", input: {...} }
{ type: "tool_result", id: "...", output: "..." }
```

This is a generic format. It does not match any specific API directly -- you translate to/from the API format at the boundary. This keeps your session representation clean and portable.

## 1. Initialize a Session

Create a session with a system prompt and the user's message:

```
function init_session(system_prompt, user_prompt):
    return {
        messages: [
            { role: "system", content: system_prompt },
            { role: "user",   content: user_prompt },
        ]
    }
```

## 2. Convert to API Format

Most model APIs expect messages grouped by role, with tool calls and tool results packed into specific block structures. Write a conversion function that:

- Skips system messages (pass them separately)
- Groups consecutive `thinking`, `assistant`, and `tool_call` messages into a single assistant message with multiple content blocks
- Groups consecutive `tool_result` messages into a single user message with multiple content blocks

```
function to_api_messages(messages):
    api_messages = []
    assistant_blocks = []
    tool_result_blocks = []

    -- When hitting a boundary (user message, or switching between
    -- assistant-side and tool-result-side), flush the accumulated
    -- blocks into a single message.

    for each message:
        if system: skip
        if user: flush both buffers, add user message
        if assistant: flush tool results, add text block to assistant buffer
        if thinking: flush tool results, add thinking block to assistant buffer
        if tool_call: flush tool results, add tool_use block to assistant buffer
        if tool_result: flush assistant buffer, add to tool result buffer

    flush remaining buffers
    return api_messages
```

The key insight is that what your session stores as separate messages may need to be combined into a single API message with multiple content blocks.

## 3. Parse the Response

The model returns a response with content blocks. Parse each block back into your generic message format:

```
function parse_response(api_response):
    messages = []
    for each block in api_response.content:
        if block is thinking: append { type: "thinking", content: block.thinking }
        if block is text:     append { role: "assistant", content: block.text }
        if block is tool_use: append { type: "tool_call", name: ..., id: ..., input: ... }
    return messages
```

## 4. Execute Tool Calls

Tool handlers are just functions. Map tool names to handler functions, then execute them:

```
function execute_tool_calls(tool_calls, tool_handlers):
    results = []
    for each tool_call:
        handler = tool_handlers[tool_call.name]
        try:
            output = handler(tool_call.input)
        catch error:
            output = "Error: " + error.message
        results.append({ type: "tool_result", id: tool_call.id, output: output })
    return results
```

Tool calls within a single response are independent of each other, so you can execute them in parallel (threads, promises, goroutines, etc).

## 5. The Loop

The agent loop ties everything together. Each iteration: call the model, add its messages to the session, check for tool calls, execute them, add results, repeat.

```
function agent_loop(invoke_model, tools, session, tool_handlers):
    loop:
        api_messages = to_api_messages(session.messages)
        api_response = invoke_model(tools, api_messages)

        new_messages = parse_response(api_response)
        append new_messages to session

        tool_calls = filter new_messages for type "tool_call"
        if no tool_calls: break  -- model is done

        results = execute_tool_calls(tool_calls, tool_handlers)
        append results to session

    return session
```

That's the entire loop. The model drives the control flow -- it decides when to call tools and when to stop (by responding with just text and no tool calls).

## 6. Defining Tools

Tools are defined as schemas that tell the model what's available. Each tool has a name, description, and input schema:

```
{
    name: "read_file",
    description: "Read the contents of a file at the given path.",
    input_schema: {
        type: "object",
        properties: {
            path: { type: "string", description: "File path to read" }
        },
        required: ["path"]
    }
}
```

Pass the tool definitions to the model on each call. The corresponding handler is just a function that takes the input and returns a string:

```
function read_file(input):
    return file_system.read(input.path)
```

## 7. Subagents

An agent can use other agents as tools. A subagent is just a tool handler that runs its own agent loop:

```
function summarize(input):
    sub_session = init_session(
        "You are a summarizer. Output only the summary.",
        input.text
    )
    result = agent_loop(invoke_model, [], sub_session, {})
    return response(result).content
```

Register it as a tool handler like any other function. The outer agent calls "summarize" as a tool, the handler spins up an inner agent loop, and returns the result. The outer agent sees a simple string tool result and has no idea a whole agent loop ran inside.

This composes naturally:
- A coordinator agent can dispatch to multiple specialist subagents
- Subagents can have their own tools (or no tools at all)
- Subagents can have different system prompts, models, or iteration limits
- You can run multiple subagents in parallel when the tool calls are independent

## 8. Context Management

Sessions grow over time. Old thinking blocks and tool call inputs can be large and become less relevant as the conversation progresses. You can compact them:

- Walk the message list backwards, counting assistant responses
- Once N assistant responses have appeared after a message, truncate its content to a short prefix
- Only compact thinking and tool_call messages -- keep user messages, assistant text, and tool results intact
- Mark compacted messages so you don't re-process them

This keeps the session under control during long-running agent loops without losing the structural flow of the conversation.

## Putting It Together

The full implementation is around 200 lines in most languages. The essential pieces:

| Component | Purpose |
|---|---|
| Session | Ordered list of generic messages |
| `to_api_messages` | Translate generic messages to API format |
| `parse_response` | Translate API response back to generic messages |
| `execute_tool_calls` | Run tool handlers, collect results |
| `agent_loop` | The loop: call model, execute tools, repeat |

Everything else -- logging, session forking, compaction, subagents -- is layered on top of these five pieces.

The model is the control flow. Your code just provides the loop scaffold and tool execution. The model decides what to do, when to use tools, and when to stop.
