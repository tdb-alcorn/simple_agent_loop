# session is in openai api spec format

def init_session(system_prompt, user_prompt):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }

def extend_session(session, message):
    session["messages"].append(message)

def send(session, user_message):
    extend_session(session, {"role": "user", "content": user_message})

def fork_session(session):
    import copy
    return copy.deepcopy(session)

def response(session):
    if len(session["messages"]) == 0:
        return None
    return session["messages"][-1]

def log(message):
    # TODO
    # Format: {iso datetime} {agent name} {role} {content, one line}
    # Truncate whole thing to be within 120 chars
    # We want to log everything except streaming chunks. Messages like thinking and tool calls and results need
    # special logic to extract the actual content and display it.
    pass

def agent_loop(invoke_model, tools, session, name=None, max_iterations=None):
    # TODO
    # while loop, invoke model with session and tools, get responses and extend session, handle tool calls and extend session with results, loop. break when model response has no tool calls, or exceed max_iterations.
    # always call tools within a try except, send back error as result if encountered
    # execute tool calls in parallel, async, background to speed up. model will send multiple tool calls in one response sometimes.
    pass
