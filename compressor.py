import anthropic
import json
from dotenv import load_dotenv
from agent_loop import init_session, agent_loop, response

load_dotenv()

client = anthropic.Anthropic()


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


# --- Subagent: Shortener ---

def shorten(text: str) -> str:
    """Shorten the given text to roughly half its length while preserving meaning.
    Returns JSON with shortened_text and compression_ratio."""
    session = init_session(
        system_prompt=(
            "You are a text compressor. Rewrite the user's text to be roughly half "
            "as long while preserving all key meaning. Output ONLY the shortened plain text, "
            "no markdown, no headers, no bullet points, no commentary. Just the compressed "
            "prose paragraph."
        ),
        user_prompt=text,
    )
    result = agent_loop(invoke_model, [], session, name="shortener", max_iterations=1)
    shortened_text = response(result)["content"]
    ratio = len(shortened_text) / len(text) if text else 0.0
    return json.dumps({"compression_ratio": round(ratio, 3), "shortened_text": shortened_text})


# --- Subagent: Judge ---

def judge(original: str, shortened: str) -> str:
    """Compare original and shortened text. Return JSON verdict."""
    session = init_session(
        system_prompt=(
            "You are a compression quality judge. The user will give you an original "
            "text and a shortened version. Compare them and return ONLY a JSON object: "
            '{"verdict": "acceptable", "reason": "..."} or '
            '{"verdict": "too_lossy", "reason": "..."}. '
            "Verdict should be 'acceptable' if key meaning is preserved, 'too_lossy' "
            "if important information was lost."
        ),
        user_prompt=f"ORIGINAL:\n{original}\n\nSHORTENED:\n{shortened}",
    )
    result = agent_loop(invoke_model, [], session, name="judge", max_iterations=1)
    return response(result)["content"]


# --- Coordinator ---

coordinator_tools = [
    {
        "name": "shorten",
        "description": (
            "Shorten the given text to roughly half its length while preserving meaning. "
            'Returns JSON with "shortened_text" and "compression_ratio" (0.0-1.0).'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to shorten"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "judge",
        "description": (
            "Compare original text against a shortened version. "
            "Returns JSON with verdict ('acceptable' or 'too_lossy') and reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "original": {"type": "string", "description": "The original full text"},
                "shortened": {"type": "string", "description": "The shortened version to evaluate"},
            },
            "required": ["original", "shortened"],
        },
    },
]

COORDINATOR_SYSTEM = """\
You are a text compression coordinator. Your goal is to iteratively compress the \
user's text as much as possible without losing key meaning.

You MUST follow this procedure strictly, step by step. Do NOT skip any step.

Step 1: Call the `shorten` tool with the current text. It returns JSON with \
"shortened_text" and "compression_ratio".
Step 2: Call the `judge` tool with the ORIGINAL user text (always the very first input) \
and the shortened_text from step 1.
Step 3: Check the results:
  - If the judge says "too_lossy": STOP and output the last acceptable version as plain text.
  - If the compression_ratio from shorten is > 0.9 (diminishing returns): \
STOP and output the shortened_text as plain text.
  - If "acceptable" and compression_ratio <= 0.9: go back to Step 1 using the \
shortened_text as the new input.

CRITICAL: You must ALWAYS call `judge` after every `shorten`. Never skip it.
When you stop, output ONLY the final compressed text with zero commentary."""


if __name__ == "__main__":
    sample_text = (
        "The Amazon rainforest, often referred to as the 'lungs of the Earth,' is a "
        "vast tropical rainforest located in South America that spans across nine "
        "countries, including Brazil, Peru, and Colombia. It covers approximately 5.5 "
        "million square kilometers, making it the largest tropical rainforest in the "
        "world. The rainforest is home to an incredibly diverse array of wildlife, "
        "including approximately 10 percent of all species known to science. It contains "
        "around 390 billion individual trees divided into roughly 16,000 different "
        "species. The Amazon River, which flows through the rainforest, is the second "
        "longest river in the world and carries more water than any other river on Earth. "
        "Deforestation poses a significant threat to the Amazon, with large areas being "
        "cleared for cattle ranching, soybean farming, and logging operations. Scientists "
        "warn that continued deforestation could push the Amazon past a tipping point, "
        "transforming large portions of the rainforest into savanna, which would have "
        "devastating consequences for global climate patterns and biodiversity."
    )

    print("ORIGINAL TEXT:")
    print(sample_text)
    print(f"\nOriginal length: {len(sample_text)} chars")
    print("=" * 80)

    session = init_session(
        system_prompt=COORDINATOR_SYSTEM,
        user_prompt=sample_text,
    )
    result = agent_loop(
        invoke_model,
        coordinator_tools,
        session,
        tool_handlers={"shorten": shorten, "judge": judge},
        name="coordinator",
    )

    final = response(result)["content"]
    print("=" * 80)
    print("\nFINAL COMPRESSED TEXT:")
    print(final)
    print(f"\nFinal length: {len(final)} chars ({len(final)*100//len(sample_text)}% of original)")
