import anthropic
from simple_agent_loop import *
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

def invoke_model(tools, session):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        messages=session["messages"],
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        tools=tools,
    )
    return response.to_dict()


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
