"""
reflection_agent.py — Enhanced Reflection Agent (LangGraph 1.x)

Concepts covered (vs. the basic MessageGraph version):
  1. StateGraph with a custom TypedDict state (instead of deprecated MessageGraph)
  2. Two reducers: add_messages (for chat) and operator.add (for a score list)
  3. Plain overwrite fields (iteration, best_score, best_draft)
  4. Structured output from the critic (numeric score)
  5. Quality-based stopping + max-iteration safety, with a path map
  6. Role flipping / context trimming for the critic
  7. Checkpointer (InMemorySaver) + thread_id -> inspect state & history
  8. Streaming (stream_mode="updates") to watch each node as it runs
  9. recursion_limit as a last-resort safety net
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from reflection_chains import Critique, generation_chain, reflection_chain

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GENERATE = "generate"
REFLECT = "reflect"

MAX_ITERATIONS = 4   # max number of drafts
TARGET_SCORE = 8     # stop early once a draft scores at least this


# ---------------------------------------------------------------------------
# 1. State: the "whiteboard" every node reads and updates
# ---------------------------------------------------------------------------
class ReflectionState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # appended (chat log)
    score_history: Annotated[list[int], operator.add]     # appended (plain list)
    iteration: int                                        # overwritten
    best_score: int                                       # overwritten
    best_draft: str                                       # overwritten


# ---------------------------------------------------------------------------
# 2. Nodes: each returns ONLY the keys it changes
# ---------------------------------------------------------------------------
def generate_node(state: ReflectionState) -> dict:
    """Write (or rewrite) the tweet using the full conversation so far."""
    draft = generation_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [draft],                    # add_messages appends it
        "iteration": state["iteration"] + 1,    # overwrite
    }


def reflect_node(state: ReflectionState) -> dict:
    """Grade the latest draft and send feedback back to the generator."""
    original_request = state["messages"][0].content
    latest_draft = state["messages"][-1].content

    # Role flipping + context trimming: the critic sees the tweet as something a
    # USER submitted for review, and only the latest draft (not the whole history).
    critic_input = [
        HumanMessage(
            content=f"Original request: {original_request}\n\n"
            f"Tweet to grade:\n{latest_draft}"
        )
    ]
    critique: Critique = reflection_chain.invoke({"messages": critic_input})

    feedback = (
        f"Score: {critique.score}/10\n"
        f"Strengths: {'; '.join(critique.strengths)}\n"
        f"Weaknesses: {'; '.join(critique.weaknesses)}\n"
        f"Suggestions: {critique.suggestions}"
    )

    update = {
        # Critique goes back as a HumanMessage so the generator treats it as user feedback
        "messages": [HumanMessage(content=feedback, name="critic")],
        "score_history": [critique.score],      # operator.add appends it
    }
    if critique.score > state["best_score"]:    # keep the best draft, not just the last
        update["best_score"] = critique.score
        update["best_draft"] = latest_draft
    return update


# ---------------------------------------------------------------------------
# 3. Router: decides where to go after reflection
# ---------------------------------------------------------------------------
def should_continue(state: ReflectionState) -> Literal["generate", "__end__"]:
    if state["score_history"][-1] >= TARGET_SCORE:
        return END        # good enough -> stop early
    if state["iteration"] >= MAX_ITERATIONS:
        return END        # out of attempts -> stop
    return GENERATE       # otherwise revise


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    builder = StateGraph(ReflectionState)
    builder.add_node(GENERATE, generate_node)
    builder.add_node(REFLECT, reflect_node)

    builder.add_edge(START, GENERATE)
    builder.add_edge(GENERATE, REFLECT)  # every draft is graded
    builder.add_conditional_edges(
        REFLECT,
        should_continue,
        {GENERATE: GENERATE, END: END},  # path map -> correct diagram + validation
    )
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------
def main(topic: str = "AI Agents taking over content creation"):
    app = build_graph(checkpointer=InMemorySaver())
    print(app.get_graph().draw_mermaid())

    config = {
        "configurable": {"thread_id": "tweet-1"},  # which "conversation" to save under
        "recursion_limit": 20,                     # safety net against infinite loops
    }
    inputs = {
        "messages": [HumanMessage(content=topic)],
        "iteration": 0,
        "best_score": 0,
        "best_draft": "",
        "score_history": [],
    }

    # Stream: see each node's update as soon as it finishes
    for update in app.stream(inputs, config, stream_mode="updates"):
        for node_name, changes in update.items():
            last_msg = changes["messages"][-1]
            print(f"\n=== {node_name.upper()} ===")
            print(last_msg.content)

    # Read the final saved state from the checkpointer
    final = app.get_state(config).values
    print("\n" + "=" * 60)
    print(f"Drafts written : {final['iteration']}")
    print(f"Score history  : {final['score_history']}")
    print(f"Best score     : {final['best_score']}/10")
    print(f"Best tweet     :\n{final['best_draft']}")

    # Every step was checkpointed — you can look back through history
    print("\nCheckpoint history (newest first):")
    for snapshot in app.get_state_history(config):
        print(f"  next={snapshot.next}  iteration={snapshot.values.get('iteration')}")


if __name__ == "__main__":
    main()