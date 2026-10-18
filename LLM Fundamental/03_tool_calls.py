"""
03_tool_calls.py
----------------
Tool calling (function calling) with the OpenAI Responses API.

Run from the project root:
    python "LLM Fundamental/03_tool_calls.py"

LEARNING MAP
    The big idea   The model NEVER runs your code. It only asks you to run a function
                   (a name + JSON arguments). You run it and send the result back.
    Part 1         Real Python functions (the tools themselves)
    Part 2         Describing tools to the model (schemas, strict mode, Pydantic)
    Part 3         Safe execution (validation, unknown tools, errors)
    Part 4         The agent loop (multi-step, parallel calls, max steps, approval)
    Part 5         Demos 0-9, one per concept

Each demo is a function. The DEMOS list at the bottom controls which ones run.
Comment a line out with Ctrl + / to skip it (most demos make 1-4 API calls).
"""

import ast
import json
import operator
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools when you need real data or "
    "to do a calculation. If no tool is needed, answer directly. "
    "Never guess a tool's result."
)


# ============================================================================
# PART 1. The actual Python functions (the "tools")
# ============================================================================
# 1a. A lookup tool. It returns an error dict for unknown cities, so we can see
#     how the model reacts to failure (demo 7).
WEATHER_DATA = {
    "delhi": {"temperature_c": 34, "condition": "hazy"},
    "mumbai": {"temperature_c": 31, "condition": "humid"},
    "bengaluru": {"temperature_c": 24, "condition": "cloudy"},
    "bangalore": {"temperature_c": 24, "condition": "cloudy"},
    "pune": {"temperature_c": 28, "condition": "sunny"},
}


def get_weather(city: str, unit: str) -> dict:
    data = WEATHER_DATA.get(city.strip().lower())
    if data is None:
        return {"error": f"No weather data for '{city}'"}
    temp = data["temperature_c"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
    return {"city": city, "temperature": temp, "unit": unit, "condition": data["condition"]}


# 1b. A calculation tool. LLMs are unreliable at arithmetic, so give them a calculator.
#     NEVER use eval() on model-written text. This walks the expression tree and only
#     allows numbers and basic operators.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> dict:
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _eval_node(tree.body)}


# 1c. A retrieval tool: a tiny in-memory "knowledge base". This is the same idea as RAG,
#     just with the retrieval step exposed to the model as a tool.
NOTES = [
    {"id": 1, "title": "Project Alpha deadline", "text": "Project Alpha delivery is due on 30 November."},
    {"id": 2, "title": "Team offsite", "text": "The team offsite is in Goa on 12 December."},
    {"id": 3, "title": "Budget review", "text": "Q4 budget review meeting is on 5 November with finance."},
    {"id": 4, "title": "Vendor decision", "text": "Vendor B was chosen for the data platform contract."},
]


def search_notes(query: str, top_k: int) -> dict:
    words = set(query.lower().split())
    scored = []
    for note in NOTES:
        haystack = (note["title"] + " " + note["text"]).lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, note))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return {"matches": [note for _, note in scored[:top_k]]}


# 1d. A tool with a SIDE EFFECT (it changes the outside world). Simulated here, but treat
#     any real "send / delete / pay / write" tool like this one: ask a human first (demo 9).
def send_email(to: str, subject: str, body: str) -> dict:
    print(f"   (simulated) email to={to} subject={subject!r}")
    return {"status": "sent (simulated)", "to": to, "subject": subject}


# ============================================================================
# PART 2. Describing the tools to the model
# ============================================================================
# The model never sees your Python code. It only sees: name + description + parameter schema.
# The DESCRIPTION is the most important part. It is how the model decides WHEN to call a tool.

# 2a. The manual way (what you wrote originally). Fine for learning, but easy to get wrong.
MANUAL_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a given city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city to get weather for"}},
        "required": ["city"],
    },
}


# 2b. The better way: describe the arguments with Pydantic (same idea as file 02),
#     and let the SDK generate a STRICT schema. Strict mode guarantees the model's
#     arguments always match the schema (right names, right types, nothing missing).
class WeatherArgs(BaseModel):
    city: str = Field(description="City name, e.g. 'Pune'")
    unit: Literal["celsius", "fahrenheit"] = Field(description="Temperature unit")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '28 * 9 / 5 + 32'")


class SearchNotesArgs(BaseModel):
    query: str = Field(description="Keywords to look for in the user's notes")
    top_k: int = Field(description="How many notes to return, between 1 and 5")


class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Plain-text email body")


def make_tool(args_model: type[BaseModel], name: str, description: str) -> dict:
    """Pydantic model -> strict tool definition in the Responses API format."""
    spec = pydantic_function_tool(args_model, name=name, description=description)["function"]
    return {"type": "function", **spec}


# 2c. The registry connects a tool NAME (what the model says) to the real function
#     and the Pydantic model used to validate its arguments.
TOOL_REGISTRY = {
    "get_weather": (get_weather, WeatherArgs, "Get the current weather for a city. Use for any question about weather or temperature."),
    "calculate": (calculate, CalculatorArgs, "Evaluate an arithmetic expression exactly. Use for any math instead of calculating yourself."),
    "search_notes": (search_notes, SearchNotesArgs, "Search the user's personal notes (deadlines, meetings, decisions). Use when the question is about the user's own information."),
    "send_email": (send_email, SendEmailArgs, "Send an email on the user's behalf. Only use when the user explicitly asks to send one."),
}

TOOLS = [make_tool(model, name, desc) for name, (_, model, desc) in TOOL_REGISTRY.items()]

# Tools with side effects that need a human "yes" before they run (demo 9).
REQUIRES_APPROVAL = {"send_email"}


# ============================================================================
# PART 3. Running tools safely
# ============================================================================
def execute_tool(name: str, arguments_json: str) -> str:
    """
    Run a tool the model asked for. ALWAYS returns a JSON string, never raises.
    Errors are sent back to the model as data, so it can recover or explain.
    """
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    func, args_model, _ = entry

    # Validate arguments before using them. The model's output is untrusted input.
    try:
        args = args_model.model_validate_json(arguments_json)
    except ValidationError as err:
        return json.dumps({"error": "Invalid arguments", "details": str(err)})

    try:
        result = func(**args.model_dump())
    except Exception as err:  # a crashing tool should not crash the agent
        return json.dumps({"error": f"Tool failed: {type(err).__name__}: {err}"})

    return json.dumps(result, default=str)


def get_function_calls(response) -> list:
    """
    Pull the tool requests out of a response. response.output is a LIST of items of
    different types ('message', 'function_call', 'reasoning', ...). Never assume
    output[0] is the tool call, and never assume there is only one.
    """
    return [item for item in response.output if item.type == "function_call"]


def approve(call) -> bool:
    """Human-in-the-loop gate for risky tools."""
    print(f"\n⚠️  The model wants to run: {call.name}({call.arguments})")
    return input("   Approve? [y/N]: ").strip().lower() == "y"


# ============================================================================
# PART 4. The agent loop
# ============================================================================
def run_agent(history: list, max_steps: int = 6, verbose: bool = True, **create_kwargs) -> str:
    """
    The core agent pattern:

        call model -> did it ask for tools?
              no  -> done, return the answer
              yes -> run the tools, append results to history, call the model again

    `history` is modified in place, so you can reuse it for a multi-turn chat.
    `max_steps` stops runaway loops (a model that keeps calling tools forever costs money).
    Extra keyword args (tool_choice, parallel_tool_calls, ...) go straight to the API.
    """
    for step in range(1, max_steps + 1):
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=history,
            tools=TOOLS,
            **create_kwargs,
        )

        # Keep EVERYTHING the model produced, in order. The API needs the original
        # function_call items in the history to match them with their outputs.
        history += response.output

        calls = get_function_calls(response)
        if not calls:
            if verbose:
                print(f"   [step {step}] no tool calls -> final answer")
            return response.output_text

        if verbose:
            print(f"   [step {step}] model requested {len(calls)} tool call(s)")

        for call in calls:
            if call.name in REQUIRES_APPROVAL and not approve(call):
                output = json.dumps({"error": "The user declined this action."})
            else:
                output = execute_tool(call.name, call.arguments)

            if verbose:
                print(f"   🔧 {call.name}({call.arguments})")
                print(f"   📦 {output}")

            # call_id links this result to the exact request it answers.
            history.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    return f"Stopped after {max_steps} steps without a final answer."


# ============================================================================
# PART 5. Demos
# ============================================================================
def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---- 0. The problem: the model has no real-time data ------------------------
def demo_0_the_problem():
    result = client.responses.create(model=MODEL, input="What is the current weather in Bangalore?")
    print(result.output_text)
    print("\nWithout tools, the model can only apologise or guess.")


# ---- 1. What does the model actually receive? (offline) ---------------------
def demo_1_inspect_tool_definitions():
    print("Manual definition (your original):")
    print(json.dumps(MANUAL_WEATHER_TOOL, indent=2))
    print("\nGenerated from Pydantic (note 'strict': True and additionalProperties=false):")
    print(json.dumps(TOOLS[0], indent=2))
    print("\nAll tools the model can see:")
    for tool in TOOLS:
        print(f" - {tool['name']}: {tool['description']}")
    print("\nTool definitions count as input tokens on every call. Keep descriptions short but clear.")


# ---- 2. One round trip, step by step (your original flow, made robust) ------
def demo_2_single_round_trip():
    history = [{"role": "user", "content": "What is the current weather in Delhi? Use celsius."}]

    # Call 1: the model decides whether to use a tool.
    step1 = client.responses.create(model=MODEL, input=history, tools=TOOLS)
    print("Output item types:", [item.type for item in step1.output])

    calls = get_function_calls(step1)
    if not calls:
        print("No tool call. Direct answer:", step1.output_text)
        return

    call = calls[0]
    print("Tool requested :", call.name)
    print("Arguments (str):", call.arguments)  # arguments arrive as a JSON STRING
    print("call_id        :", call.call_id)

    # We run the function. The model never does.
    output = execute_tool(call.name, call.arguments)
    print("📦 Raw result  :", output)

    # Call 2: send the original items plus our result back.
    history += step1.output
    history.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    step2 = client.responses.create(model=MODEL, input=history, tools=TOOLS)
    print("\n💬 Final answer:", step2.output_text)


# # ---- 3. The model can decide NOT to use a tool ------------------------------
def demo_3_no_tool_needed():
    history = [{"role": "user", "content": "What is the capital of France?"}]
    response = client.responses.create(model=MODEL, input=history, tools=TOOLS)
    print("Tool calls requested:", [c.name for c in get_function_calls(response)])
    print("Answer:", response.output_text)
    print("\nTools are OPTIONS, not obligations. Good descriptions tell the model when they apply.")


# # ---- 4. Parallel tool calls -------------------------------------------------
def demo_4_parallel_calls():
    question = "Compare the weather in Delhi, Mumbai and Bengaluru in celsius."

    print("A) Parallel allowed (default): look at how many calls arrive in ONE step")
    print(run_agent([{"role": "user", "content": question}]))

    print("\nB) parallel_tool_calls=False: one call per step")
    print(run_agent([{"role": "user", "content": question}], parallel_tool_calls=False))


# # ---- 5. Multi-step reasoning: one tool's output feeds the next --------------
def demo_5_agent_loop():
    question = (
        "Get the temperature in Pune in celsius, then use the calculator tool "
        "to convert that temperature to Fahrenheit."
    )
    print(run_agent([{"role": "user", "content": question}]))
    print("\nThe second call needed the FIRST result, so a single round trip can't do this.")
    print("This loop is the heart of every agent framework, including LangGraph.")


# # ---- 6. Controlling tool use with tool_choice -------------------------------
def demo_6_tool_choice():
    question = [{"role": "user", "content": "Tell me a fun fact about Delhi."}]
    choices = [
        "auto",  # the model decides (default)
        "none",  # tools disabled for this call
        "required",  # the model must call some tool
        {"type": "function", "name": "get_weather"},  # the model must call THIS tool
    ]
    for choice in choices:
        response = client.responses.create(model=MODEL, input=question, tools=TOOLS, tool_choice=choice)
        names = [c.name for c in get_function_calls(response)]
        print(f"tool_choice={str(choice):<55} -> {names}")


# # ---- 7. Errors: tools fail, models send bad arguments -----------------------
def demo_7_error_handling():
    print("Part A (offline): execute_tool never raises, it returns errors as data\n")
    bad_calls = [
        ("unknown tool", "launch_rockets", "{}"),
        ("broken JSON", "get_weather", "{not valid json"),
        ("missing field", "get_weather", '{"city": "Pune"}'),
        ("unsafe expression", "calculate", "{\"expression\": \"__import__('os').system('dir')\"}"),
        ("division by zero", "calculate", '{"expression": "10 / 0"}'),
    ]
    for label, name, args in bad_calls:
        print(f"{label:<18} -> {execute_tool(name, args)[:110]}")

    print("\nPart B (online): the tool returns an error, the model explains it")
    print(run_agent([{"role": "user", "content": "What is the weather in Atlantis, in celsius?"}]))


# # ---- 8. Multiple tools: the model chooses the right one ---------------------
def demo_8_choosing_between_tools():
    questions = [
        "When is the Project Alpha deadline?",
        "What is 15% of 2480?",
        "Is it sunny in Pune right now? Use celsius.",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print("A:", run_agent([{"role": "user", "content": q}], verbose=False))


# # ---- 9. Human approval for side effects (interactive) -----------------------
def demo_9_human_approval():
    question = "Send an email to priya@example.com with subject 'Report' saying the report is ready."
    print(run_agent([{"role": "user", "content": question}]))
    print("\nRun it twice: once answering y, once answering n, and compare the final answers.")


# # ---- 10. Multi-turn conversation with memory --------------------------------
def demo_10_conversation_memory():
    history = []  # one list shared across turns
    turns = [
        "What's the weather in Delhi? Use celsius.",
        "And in Mumbai?",
        "Which of the two is hotter?",  # answerable from history, no tool needed
    ]
    for turn in turns:
        print(f"\n👤 {turn}")
        history.append({"role": "user", "content": turn})
        print("💬", run_agent(history))
    print(f"\nHistory now holds {len(history)} items: messages, tool requests and tool results.")


# # ============================================================================
# # Run the demos. Comment out any line (Ctrl + /) to skip it.
# # ============================================================================
DEMOS = [
    demo_0_the_problem,
    demo_1_inspect_tool_definitions,  # offline
    demo_2_single_round_trip,
    demo_3_no_tool_needed,
    demo_4_parallel_calls,
    demo_5_agent_loop,
    demo_6_tool_choice,
    demo_7_error_handling,
    demo_8_choosing_between_tools,
    demo_10_conversation_memory,
    demo_9_human_approval,  # interactive: asks for y/n, so it runs last
]


def main():
    for demo in DEMOS:
        header(demo.__name__)
        try:
            demo()
        except Exception as err:  # one failing demo should not stop the rest
            print(f"Demo failed: {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()