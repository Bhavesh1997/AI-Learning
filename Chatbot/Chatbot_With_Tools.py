"""
chatbot_with_tools.py — A tool-using chatbot built with LangGraph (ReAct pattern)

What this chatbot can do:
  - Chat normally, answering from the LLM's own knowledge
  - Search the web (Tavily) when it needs fresh information
  - Tell the current date/time (a custom tool we write ourselves)
  - REMEMBER the conversation across turns (checkpointer + thread_id)
  - Show which tools it is using while it works (streaming)

Concepts covered:
   1. State with the add_messages reducer
   2. Ready-made tools (Tavily) and custom tools (@tool)
   3. bind_tools: telling the LLM which tools exist
   4. System prompt added at call time (not stored in memory)
   5. The ReAct loop: chatbot -> tools -> chatbot -> ... -> END
   6. A custom router vs the prebuilt tools_condition
   7. ToolNode with error handling
   8. Memory: checkpointer + thread_id
   9. Streaming node updates to show tool usage
  10. recursion_limit as a safety net

Setup:
  pip install -U langgraph langchain-openai langchain-tavily python-dotenv
  .env file:
      OPENAI_API_KEY=sk-...
      TAVILY_API_KEY=tvly-...      (free key at tavily.com)
"""

from datetime import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load API keys from .env BEFORE creating any LLM or tool objects,
# because ChatOpenAI and TavilySearch read their keys when they are created.
load_dotenv()


# =============================================================================
# 1. STATE: the shared "whiteboard" for the graph
# =============================================================================
class ChatState(TypedDict):
    # add_messages is the reducer: every message a node returns is APPENDED
    # to this list (or replaces an existing message with the same ID).
    # Without it, each node would overwrite the whole conversation.
    #
    # Note: LangGraph also ships a ready-made version of this exact class:
    #     from langgraph.graph import MessagesState
    messages: Annotated[list, add_messages]


# =============================================================================
# 2. TOOLS: functions the LLM can ask us to run
# =============================================================================

# --- 2a. A ready-made tool: Tavily web search ---------------------------------
# Returns the top results for a query. The LLM decides WHEN to search and
# WHAT to search for; it fills in the "query" argument itself.
search_tool = TavilySearch(max_results=2)


# --- 2b. A custom tool: written by us with the @tool decorator ----------------
# Why this tool? An LLM has no clock. Ask it "what's today's date?" and it will
# guess from its training data. This tool gives it the real date.
#
# How @tool works:
#   - function name   -> tool name the LLM sees ("get_current_datetime")
#   - docstring       -> tool description; the LLM reads this to decide when
#                        to use the tool, so write it clearly
#   - type hints      -> argument schema (this tool takes no arguments)
@tool
def get_current_datetime() -> str:
    """Get the current date and time. Use this whenever the user asks about
    today's date, the current time, or anything relative to 'now'
    (for example 'this week' or 'latest')."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


tools = [search_tool, get_current_datetime]


# =============================================================================
# 3. LLM: OpenAI model with the tools attached
# =============================================================================
# temperature=0 keeps tool decisions consistent and predictable.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# bind_tools converts each tool into a JSON schema (name + description +
# arguments) and sends it with EVERY request. The LLM can then reply with
# either normal text OR a structured tool call such as:
#     {"name": "tavily_search", "args": {"query": "..."}, "id": "call_123"}
# The LLM never runs the tool itself; it only requests it.
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# 4. SYSTEM PROMPT: the chatbot's standing instructions
# =============================================================================
# We add this at call time instead of storing it in the state.
# Why? The state is saved by the checkpointer. If we stored the system message
# there, we would have to make sure it is added exactly once, and changing the
# prompt later would not affect old conversations. Prepending it on every call
# keeps the saved history clean: only user, AI and tool messages.
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant with access to tools.\n"
        "- Use tavily_search for current events, recent facts, or anything you are unsure about.\n"
        "- Use get_current_datetime when the question depends on today's date or time.\n"
        "- Answer from your own knowledge when no tool is needed.\n"
        "- When you use search results, mention the source website.\n"
        "- Keep answers concise."
    )
)


# =============================================================================
# 5. NODES
# =============================================================================
def chatbot(state: ChatState) -> dict:
    """REASON step: the LLM reads the whole conversation (including any
    tool results) and decides to either call tools or answer the user."""
    messages_for_llm = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages_for_llm)
    # Return only what changed. add_messages appends this AIMessage.
    return {"messages": [response]}


# ACT step: ToolNode finds the most recent AIMessage, runs EVERY tool call in
# it (in parallel), and returns one ToolMessage per call, linked by
# tool_call_id.
#
# handle_tool_errors: by default, ToolNode only reports the LLM's own mistakes
# (unknown tool name, wrong arguments) back as messages. Any exception raised
# INSIDE a tool (network error, bad API key, ...) would crash the whole graph.
# With a message here, the error goes back to the LLM as a ToolMessage with
# status="error", so the LLM can explain the problem or try another approach.
tool_node = ToolNode(
    tools,
    handle_tool_errors="The tool failed to run. Tell the user the tool is "
    "unavailable right now and answer from your own knowledge if you can.",
)


# =============================================================================
# 6. ROUTER: decides what happens after the chatbot node
# =============================================================================
def tools_router(state: ChatState) -> str:
    """If the LLM's last message asked for tools, go run them.
    Otherwise the LLM has given its final answer, so stop."""
    last_message = state["messages"][-1]

    # Only AIMessages can contain tool_calls. An empty list means no tools.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END

    # This router is exactly what the prebuilt helper does:
    #     from langgraph.prebuilt import tools_condition
    #     graph.add_conditional_edges("chatbot", tools_condition)
    # (tools_condition expects the tool node to be named "tools".)
    # Writing it yourself is useful when you need custom rules, for example
    # "ask a human before running an expensive tool".


# =============================================================================
# 7. GRAPH: wire the nodes into a loop
# =============================================================================
#   START -> chatbot --(tool calls?)--> tool_node
#               ^                          |
#               +--------------------------+   (results go back to the LLM)
#               |
#            (no tool calls)
#               v
#              END
def build_graph(checkpointer=None):
    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "chatbot")  # newer style of set_entry_point("chatbot")

    # Path map: lists every place the router may send us.
    # It validates the router's return values and draws the diagram correctly.
    graph.add_conditional_edges(
        "chatbot",
        tools_router,
        {"tool_node": "tool_node", END: END},
    )

    # The back-edge that makes this an agent: after the tools run, the
    # LLM sees their results (OBSERVE) and reasons again.
    graph.add_edge("tool_node", "chatbot")

    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 8. RUN: an interactive chat loop with memory and streaming
# =============================================================================
def print_update(node_name: str, update: dict) -> None:
    """Print a short, readable line for each step instead of the raw state."""
    for msg in update["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"   [calling tool] {call['name']}({call['args']})")
        elif isinstance(msg, ToolMessage):
            status = "error" if msg.status == "error" else "done"
            print(f"   [tool {status}] {msg.name}")
        elif isinstance(msg, AIMessage):
            print(f"\nAssistant: {msg.content}\n")


def main():
    # MEMORY: the checkpointer saves the state after every step, grouped by
    # thread_id. On the next invoke with the same thread_id, LangGraph loads
    # the saved messages first, and add_messages appends the new question.
    #
    # Without a checkpointer, every invoke starts from an EMPTY state, so the
    # bot forgets everything you said in earlier turns.
    #
    # InMemorySaver lives in RAM and is lost when the program stops.
    # For real apps use SqliteSaver or PostgresSaver (separate packages).
    app = build_graph(checkpointer=InMemorySaver())

    thread_number = 1
    print("Chatbot ready. Commands: 'new' = fresh conversation, 'exit' = quit.\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ("exit", "end", "quit"):
            break
        if user_input.lower() == "new":
            thread_number += 1  # a new thread_id means an empty conversation
            print(f"--- Started conversation {thread_number} ---\n")
            continue
        if not user_input:
            continue

        config = {
            "configurable": {"thread_id": f"chat-{thread_number}"},
            # Safety net: stop if the LLM keeps calling tools in a loop.
            # Each node run counts as one step.
            "recursion_limit": 15,
        }

        # We only send the NEW message. The checkpointer adds the history.
        # stream_mode="updates" gives us each node's output as soon as that
        # node finishes, so we can show tool calls live.
        for step in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="updates",
        ):
            for node_name, update in step.items():
                print_update(node_name, update)


if __name__ == "__main__":
    main()