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


# --- Subagent: Editor ---

def edit(text: str, rules: str, specific_info: str) -> str:
    """Apply transformation rules and specific info to source text, return transformed text."""
    session = init_session(
        system_prompt=(
            "Apply the given rules to the text, incorporating the specific info. "
            "Output ONLY the resulting text."
        ),
        user_prompt=f"SOURCE TEXT:\n{text}\n\nRULES:\n{rules}\n\nSPECIFIC INFO:\n{specific_info}",
    )
    result = agent_loop(invoke_model, [], session, name="editor", max_iterations=1)
    output = response(result)["content"]
    print(f"\n--- Editor Output ---\n{output}\n--- End Output ---\n")
    return output


# --- Subagent: Similarity Judge ---

def judge_similarity(editor_output: str, target: str) -> str:
    """Compare editor output against target text. Return JSON with score and differences."""
    session = init_session(
        system_prompt=(
            "You are a similarity judge. Compare the editor output against the target text. "
            "Return ONLY a JSON object: "
            '{"score": <0-100>, "differences": "..."} where score reflects how closely the '
            "editor output matches the target. 100 means a perfect match. "
            "Be strict: wording doesn't need to be identical, but all content, tone, "
            "structure, and formatting must match. Deduct points for any missing, extra, "
            "or mismatched content."
        ),
        user_prompt=f"EDITOR OUTPUT:\n{editor_output}\n\nTARGET:\n{target}",
    )
    result = agent_loop(invoke_model, [], session, name="similarity-judge", max_iterations=1)
    output = response(result)["content"]
    print(f"\n--- Similarity Judge ---\n{output}\n")
    return output


# --- Subagent: Generality Judge ---

def judge_generality(rules: str) -> str:
    """Check that rules contain no specific content. Return JSON with score and issues."""
    session = init_session(
        system_prompt=(
            "You are a generality judge. Check that the given transformation rules are "
            "fully general and contain NO specific content from any particular text -- "
            "no specific names, numbers, dates, quoted phrases, or other concrete details. "
            "Rules should describe transformations abstractly (e.g. 'change tone from casual "
            "to formal') not reference specific content (e.g. 'change John to Mr. Smith'). "
            "Return ONLY a JSON object: "
            '{"score": <0-100>, "issues": "..."} where score reflects how general the rules are. '
            "100 means fully abstract with zero specific content. Deduct points for each "
            "piece of specific content that leaked into the rules."
        ),
        user_prompt=f"RULES:\n{rules}",
    )
    result = agent_loop(invoke_model, [], session, name="generality-judge", max_iterations=1)
    output = response(result)["content"]
    print(f"\n--- Generality Judge ---\n{output}\n")
    return output


# --- Subagent: Specific Info Judge ---

def judge_specific_info(specific_info: str) -> str:
    """Check that specific_info is a flat list of facts. Return JSON with score and issues."""
    session = init_session(
        system_prompt=(
            "You are a specific-info judge. Check that the given specific_info is ONLY a flat "
            "list of specific facts and snippets of information -- names, dates, numbers, titles, "
            "labels, and other concrete details.\n\n"
            "Deduct points heavily for:\n"
            "- Explanations or reasoning (e.g. 'because the tone is formal, use...')\n"
            "- Formatting or structure notes (e.g. 'use a numbered list', 'add a header')\n"
            "- Verbatim text to include (e.g. 'write: Please be advised that...')\n"
            "- Instructions or directives (e.g. 'change X to Y', 'make sure to...')\n\n"
            "specific_info should contain ONLY raw facts like: "
            "'Full name: Michael Chen', 'Title: Project Lead', 'Date: March 15, 2025'. "
            "Return ONLY a JSON object: "
            '{"score": <0-100>, "issues": "..."} where 100 means a clean flat list of facts '
            "with no explanations, structure notes, or verbatim copy."
        ),
        user_prompt=f"SPECIFIC INFO:\n{specific_info}",
    )
    result = agent_loop(invoke_model, [], session, name="specific-info-judge", max_iterations=1)
    output = response(result)["content"]
    print(f"\n--- Specific Info Judge ---\n{output}\n")
    return output


# --- Coordinator ---

coordinator_tools = [
    {
        "name": "edit",
        "description": (
            "Apply transformation rules and specific info to the source text. "
            "Returns the transformed text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The source text to transform"},
                "rules": {"type": "string", "description": "General transformation rules to apply"},
                "specific_info": {"type": "string", "description": "Specific supplemental info (names, dates, facts) needed to produce the target"},
            },
            "required": ["text", "rules", "specific_info"],
        },
    },
    {
        "name": "judge_similarity",
        "description": (
            "Compare editor output against target text. "
            'Returns JSON with "score" (0-100) and "differences" (string).'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "editor_output": {"type": "string", "description": "The text produced by the editor"},
                "target": {"type": "string", "description": "The target text to compare against"},
            },
            "required": ["editor_output", "target"],
        },
    },
    {
        "name": "judge_generality",
        "description": (
            "Check that rules contain no specific content (names, numbers, quotes). "
            'Returns JSON with "score" (0-100) and "issues" (string).'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rules": {"type": "string", "description": "The transformation rules to evaluate"},
            },
            "required": ["rules"],
        },
    },
    {
        "name": "judge_specific_info",
        "description": (
            "Check that specific_info is a flat list of facts with no explanations, "
            "structure notes, or verbatim copy. "
            'Returns JSON with "score" (0-100) and "issues" (string).'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "specific_info": {"type": "string", "description": "The specific_info to evaluate"},
            },
            "required": ["specific_info"],
        },
    },
    {
        "name": "print",
        "description": "Print text to the console for the user to monitor progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to print"},
            },
            "required": ["text"],
        },
    },
]


def print_to_stdout(text: str) -> str:
    """Print text to stdout for monitoring."""
    print(f"\n{text}\n")
    return "Printed."


COORDINATOR_SYSTEM = """\
You are a transform-rule derivation agent. Given a source text and a target text, \
your goal is to derive (1) general transformation rules and (2) specific supplemental \
info that together let an editor reproduce the target from the source.

Rules must be GENERAL -- they describe abstract transformations (tone shifts, structural \
changes, formatting patterns) with NO specific content (no names, numbers, or quoted phrases). \
All specific content belongs in specific_info.

Follow this procedure strictly:

Step 1: Analyze the source and target. Draft initial `rules` (general transforms) and \
`specific_info` (concrete details needed).
Step 1b: Call `print` to display your drafted rules and specific_info so the user can monitor.
Step 2: Call `edit` with the source text, your rules, and specific_info.
Step 3: Call ALL THREE judges in parallel: \
`judge_similarity` (comparing editor output to target), \
`judge_generality` (checking your rules), and \
`judge_specific_info` (checking your specific_info).
Step 4: Check results:
  - If ALL THREE judges score above 90: STOP. Output ONLY a JSON object: \
{"rules": "...", "specific_info": "..."}
  - If ALL THREE judges score above 90 YOU MUST STOP. Do NOT continue past this point of diminishing returns.
  - If any judge scores 90 or below: refine your rules and/or specific_info based \
on the feedback, then call `print` with your updated rules and specific_info, and go back to Step 2.

CRITICAL: Always call all three judges after every edit. Never skip a judge. \
When you stop, output ONLY the final JSON object with zero commentary."""


if __name__ == "__main__":
    source = (
        "Hey team,\n\n"
        "Quick heads up -- we're doing a meeting this Thursday at 2pm in the big "
        "conference room. Sarah's gonna walk us through the Q3 numbers and Dave will "
        "give an update on the website redesign. Should take about an hour.\n\n"
        "If you can't make it, let me know and I'll send you the notes after.\n\n"
        "Cheers,\nMike"
    )

    target = (
        "MEMORANDUM\n\n"
        "TO: All Team Members\n"
        "FROM: Michael Chen, Project Lead\n"
        "DATE: March 15, 2025\n"
        "RE: Mandatory Team Meeting -- Q3 Review and Website Redesign\n\n"
        "Please be advised that a mandatory team meeting has been scheduled for "
        "Thursday, March 17, 2025, at 2:00 PM in Conference Room A.\n\n"
        "AGENDA:\n"
        "1. Q3 Financial Review -- Presented by Sarah Martinez, Finance Director\n"
        "2. Website Redesign Progress Update -- Presented by David Park, Lead Developer\n\n"
        "The meeting is expected to last approximately one hour. Attendance is required. "
        "Should you be unable to attend, please notify the undersigned in advance to "
        "arrange for distribution of meeting minutes.\n\n"
        "Regards,\n"
        "Michael Chen\n"
        "Project Lead"
    )

    print("SOURCE:")
    print(source)
    print("\n" + "=" * 80)
    print("\nTARGET:")
    print(target)
    print("\n" + "=" * 80)

    session = init_session(
        system_prompt=COORDINATOR_SYSTEM,
        user_prompt=f"SOURCE TEXT:\n{source}\n\nTARGET TEXT:\n{target}",
    )
    result = agent_loop(
        invoke_model,
        coordinator_tools,
        session,
        tool_handlers={
            "edit": edit,
            "judge_similarity": judge_similarity,
            "judge_generality": judge_generality,
            "judge_specific_info": judge_specific_info,
            "print": print_to_stdout,
        },
        name="coordinator",
        max_iterations=30,
    )

    final = response(result)
    print("\n" + "=" * 80)
    if final:
        print("\nFINAL OUTPUT:")
        print(final['content'])
    else:
        print("Max iterations hit")
