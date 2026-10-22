"""
Triage pipeline (ported from Langchain/Step_5.py).

Same LangChain logic as the terminal app, with the terminal bits removed:
- no input() loop / argparse
- the global DEBUG flag + print() tracer is replaced by a per-request trace
  list (a ContextVar), so concurrent API requests never mix their traces.
"""

import json
import os
from contextvars import ContextVar
from pathlib import Path

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")

CATEGORIES = ["Technical Support", "Billing", "General Inquiry"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
DEFAULT_SENTIMENT = {"sentiment": "Neutral", "urgency": 3}

ESCALATION_TAG = "[URGENT ESCALATION]"

# None = tracing off for this request; a list = collect trace entries into it.
_trace_sink: ContextVar[list | None] = ContextVar("trace_sink", default=None)


# ----------------------------------------------------------------------
# Tracer (Step 5 --debug, now returned in the API response)
# ----------------------------------------------------------------------
def start_trace() -> list:
    """Turn tracing on for the current request and return the list it fills."""
    sink: list = []
    _trace_sink.set(sink)
    return sink


def stop_trace() -> None:
    _trace_sink.set(None)


def trace(label: str, value) -> None:
    sink = _trace_sink.get()
    if sink is None:
        return
    sink.append({"step": label, "value": value})


def tap(label: str, pick=None):
    """Pass-through Runnable that records what flows through it."""

    def _tap(value):
        trace(label, pick(value) if pick else value)
        return value

    return RunnableLambda(_tap)


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def load_prompts() -> dict:
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


# ----------------------------------------------------------------------
# Output cleaners
# ----------------------------------------------------------------------
def normalize_category(raw: str) -> str:
    text = raw.strip().lower()
    for category in CATEGORIES:
        if category.lower() in text:
            return category
    return "General Inquiry"


def parse_sentiment(raw: str) -> dict:
    """Raw JSON string -> validated dict."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return dict(DEFAULT_SENTIMENT)
    if not isinstance(data, dict):
        return dict(DEFAULT_SENTIMENT)

    sentiment = str(data.get("sentiment", "")).strip().capitalize()
    if sentiment not in SENTIMENTS:
        sentiment = DEFAULT_SENTIMENT["sentiment"]

    try:
        urgency = int(data.get("urgency", DEFAULT_SENTIMENT["urgency"]))
    except (TypeError, ValueError):
        urgency = DEFAULT_SENTIMENT["urgency"]
    urgency = max(1, min(5, urgency))

    return {"sentiment": sentiment, "urgency": urgency}


# ----------------------------------------------------------------------
# Escalation rule
# ----------------------------------------------------------------------
def needs_escalation(sentiment: dict) -> bool:
    """Negative sentiment AND urgency of 4 or 5."""
    return sentiment["sentiment"] == "Negative" and sentiment["urgency"] >= 4


def add_escalation_tag(text: str) -> str:
    text = text.strip()
    if text.startswith(ESCALATION_TAG):
        return text
    return f"{ESCALATION_TAG}\n\n{text}"


# ----------------------------------------------------------------------
# Chain builders
# ----------------------------------------------------------------------
def make_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    block = prompts[name]
    return ChatPromptTemplate.from_messages(
        [("system", block["system"]), ("human", block["human"])]
    )


def make_draft_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompts[name]["system"]),
            MessagesPlaceholder("history"),
            ("human", "{query}"),
        ]
    )


def build_classifier_chain(prompts: dict, llm):
    return (
        make_prompt(prompts, "classifier")
        | llm
        | StrOutputParser()
        | tap("classifier raw output")
        | RunnableLambda(normalize_category)
    )


def build_sentiment_chain(prompts: dict, llm):
    return (
        make_prompt(prompts, "sentiment")
        | llm
        | StrOutputParser()
        | tap("sentiment raw JSON string")
        | RunnableLambda(parse_sentiment)
        | tap("sentiment parsed dict")
    )


def build_triage_chain(prompts: dict, llm):
    """Classifier and sentiment run in parallel on the same input."""
    return RunnableParallel(
        category=build_classifier_chain(prompts, llm),
        sentiment=build_sentiment_chain(prompts, llm),
    )


def build_router_chain(prompts: dict, llm):
    """Conditional routing. Input needs: category, history, query."""
    technical = make_draft_prompt(prompts, "technical_draft") | llm | StrOutputParser()
    billing = make_draft_prompt(prompts, "billing_draft") | llm | StrOutputParser()
    general = make_draft_prompt(prompts, "general_draft") | llm | StrOutputParser()

    def pick_technical(x):
        if x["category"] == "Technical Support":
            trace("selected routing branch", "technical_draft")
            return True
        return False

    def pick_billing(x):
        if x["category"] == "Billing":
            trace("selected routing branch", "billing_draft")
            return True
        trace("selected routing branch", "general_draft (default)")
        return False

    return RunnableBranch(
        (pick_technical, technical),
        (pick_billing, billing),
        general,
    )


def build_refiner_chain(prompts: dict, llm):
    return make_prompt(prompts, "tone_refiner") | llm | StrOutputParser()


def build_response_chain(prompts: dict, llm):
    """Sequential chaining: router -> refiner -> escalation tag."""
    router = build_router_chain(prompts, llm)
    refiner = build_refiner_chain(prompts, llm)

    def refiner_inputs(x: dict) -> dict:
        return {
            "draft": x["draft"],
            "sentiment": x["sentiment"]["sentiment"],
            "urgency": x["sentiment"]["urgency"],
            "escalate": "YES" if needs_escalation(x["sentiment"]) else "NO",
        }

    def finalize(x: dict) -> dict:
        escalate = needs_escalation(x["sentiment"])
        trace("escalation rule (Negative AND urgency>=4)", escalate)
        text = add_escalation_tag(x["polished"]) if escalate else x["polished"].strip()
        return {"reply": text, "escalated": escalate}

    return (
        RunnablePassthrough.assign(draft=router)
        | tap("raw draft response", pick=lambda x: x["draft"])
        | RunnablePassthrough.assign(
            polished=RunnableLambda(refiner_inputs) | refiner
        )
        | tap("tone-polished response", pick=lambda x: x["polished"])
        | RunnableLambda(finalize)
    )


# ----------------------------------------------------------------------
# One object the API holds for its whole lifetime
# ----------------------------------------------------------------------
class TriagePipeline:
    def __init__(self) -> None:
        prompts = load_prompts()
        llm = build_llm()
        self.triage_chain = build_triage_chain(prompts, llm)
        self.response_chain = build_response_chain(prompts, llm)

    async def run(self, query: str, history: list) -> dict:
        """Triage + route + refine one message. Returns reply and metadata."""
        triage = await self.triage_chain.ainvoke({"query": query})
        result = await self.response_chain.ainvoke(
            {
                "category": triage["category"],
                "sentiment": triage["sentiment"],
                "history": history,
                "query": query,
            }
        )
        return {
            "category": triage["category"],
            "sentiment": triage["sentiment"]["sentiment"],
            "urgency": triage["sentiment"]["urgency"],
            "escalated": result["escalated"],
            "reply": result["reply"],
        }
