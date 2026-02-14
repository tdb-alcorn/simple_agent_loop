import anthropic
from agent_loop import *
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
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    }
]

# Your tool implementation
def get_weather(location: str) -> str:
    return json.dumps({"location": location, "temperature": "68°F", "condition": "Sunny"})

# Initial request
session = {
    "messages": [
        {"role": "user", "content": "What's the weather in SF?"}
    ]
}

result = invoke_model(tools, session)

print(json.dumps(result))
