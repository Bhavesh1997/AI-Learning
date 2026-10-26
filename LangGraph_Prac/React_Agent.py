"""
React_Agent.py — a ReAct agent with real web search, built by hand (LangGraph 1.x)

ReAct = Reason + Act. The LLM loops:
    THINK  -> "what do I need to answer this?"
    ACT    -> ask for tool call(s)
    OBSERVE-> tool results are fed back to the LLM
    ... repeat until it answers in plain text.

    START -> agent --(tool calls?)--> tools -> agent -> ...
                  \--(none / limit hit)--> END

What's new vs. the basic version (and why it's good to trace in LangSmith):
  1. More tools: get_weather, celsius_to_fahrenheit, calculator, compare_temperatures
  2. Chained tools: weather -> convert C->F -> compare -> calculate
     (the output of one tool decides the next call, so the loop runs several times)
  3. A tool that can FAIL (divide by zero / bad expression) -> see error handling in the trace
  4. Richer state: iteration counter + log of tools used (operator.add reducer)
  5. Safety limit: MAX_ITERATIONS stops runaway loops
  6. LangSmith: run_name, tags, metadata on the run, and @traceable on a helper

Setup:
  .env needs: OPENAI_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_API_KEY
"""

import ast
import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable

load_dotenv()  # must run before the clients read their API keys

AGENT = "agent"
TOOLS = "tools"

MAX_ITERATIONS = 6  # max number of LLM "thinks" per run


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature in Celsius."""
    fake_weather = {          # stand-in for a real weather API
        "paris": "18°C, cloudy",
        "london": "14°C, rainy",
        "mumbai": "32°C, humid",
    }
    return fake_weather.get(city.lower(), f"No weather data for {city}")


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return round(celsius * 9 / 5 + 32, 1)


# Safe math evaluator: only allows numbers and + - * / ** (no eval() of arbitrary code)
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> float:
    """Evaluate a math expression, e.g. '(64.4 - 89.6) * 3'. Supports + - * / **."""
    return _eval(ast.parse(expression, mode="eval").body)  # can raise -> shows in trace


@traceable(name="compare_temperatures", run_type="tool")
def _compare(a: float, b: float) -> str:
    return "first is warmer" if a > b else "second is warmer" if b > a else "equal"


@tool
def compare_temperatures(first: float, second: float) -> str:
    """Say which of two temperatures is warmer (same unit for both)."""
    return _compare(first, second)  # nested traceable -> appears as a child span


tools = [get_weather, celsius_to_fahrenheit, calculator, compare_temperatures]


# ---------------------------------------------------------------------------
# 2. LLM with tools bound
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content="You are a careful assistant. Use get_weather for weather; never guess "
    "temperatures. Convert units and do maths with the tools, not in your head. "
    "Call tools one step at a time when a later call depends on an earlier result. "
    "When you have everything, answer concisely and show the numbers."
)


# ---------------------------------------------------------------------------
# 3. State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # appended (chat log)
    iteration: int                                        # overwritten
    tools_used: Annotated[list[str], operator.add]        # appended (audit log)


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """THINK: the LLM sees everything so far and asks for tools or answers."""
    response = llm.invoke([SYSTEM_PROMPT] + state["messages"])
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
        "tools_used": [c["name"] for c in response.tool_calls],
    }


# handle_tool_errors=True: a failing tool returns its error as a ToolMessage,
# so the LLM can read it and retry instead of the whole graph crashing.
tool_node = ToolNode(tools, handle_tool_errors=True)


# ---------------------------------------------------------------------------
# 5. Router
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    if state["iteration"] >= MAX_ITERATIONS:
        return END                       # safety stop
    if state["messages"][-1].tool_calls:
        return TOOLS                     # LLM asked for tools -> ACT
    return END                           # plain answer -> done


# ---------------------------------------------------------------------------
# 6. Graph
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node(AGENT, agent_node)
    builder.add_node(TOOLS, tool_node)

    builder.add_edge(START, AGENT)
    builder.add_conditional_edges(AGENT, should_continue, {TOOLS: TOOLS, END: END})
    builder.add_edge(TOOLS, AGENT)       # the loop
    return builder.compile()


# ---------------------------------------------------------------------------
# 7. Run it — config carries the LangSmith labels
# ---------------------------------------------------------------------------
QUERY = (
    "Compare the current weather in Paris and Mumbai. Convert both temperatures to "
    "Fahrenheit, tell me which city is warmer, and compute the difference in Fahrenheit "
    "multiplied by 3."
)


def main(query: str = QUERY):
    app = build_graph()

    config = {
        "run_name": "react-weather-agent",            # name shown in LangSmith
        "tags": ["react", "learning"],      # filterable tags
        "metadata": {"query_type": "multi-step-weather", "max_iterations": MAX_ITERATIONS},
        "recursion_limit": 25,                        # last-resort safety net
    }
    inputs = {"messages": [HumanMessage(content=query)], "iteration": 0, "tools_used": []}

    for update in app.stream(inputs, config, stream_mode="updates"):
        for node_name, changes in update.items():
            msg = changes["messages"][-1]
            print(f"\n=== {node_name.upper()} ===")
            if node_name == AGENT and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"ACT: {call['name']}({call['args']})")
            elif node_name == TOOLS:
                for m in changes["messages"]:
                    print(f"OBSERVE: {m.name} -> {str(m.content)[:300]}")
            else:
                print(f"FINAL ANSWER: {msg.content}")


if __name__ == "__main__":
    main()
