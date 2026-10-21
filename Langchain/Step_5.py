"""
Customer Support Ticket Triage & Response Router
------------------------------------------------
Terminal app. Assignment 1 - LangChain Fundamentals.

STEP 1: terminal loop + conversation history         (Task 1)
STEP 2: parallel triage - classifier + sentiment     (Task 2)
STEP 3: conditional routing to department drafts     (Task 3)
STEP 4: sequential tone refiner + escalation rule    (Task 4)
STEP 5: --debug intermediate pipeline tracer         (Task 5)

Usage:
    python triage_app.py
    python triage_app.py --debug
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")

CATEGORIES = ["Technical Support", "Billing", "General Inquiry"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
DEFAULT_SENTIMENT = {"sentiment": "Neutral", "urgency": 3}

ESCALATION_TAG = "[URGENT ESCALATION]"

# Set once at startup by main(); read by trace().
DEBUG = False


# ----------------------------------------------------------------------
# Step 5: the tracer
# ----------------------------------------------------------------------
def trace(label: str, value) -> None:
    """Print an intermediate pipeline value, but only when --debug is on."""
    if not DEBUG:
        return
    if isinstance(value, (dict, list)):
        value = json.dumps(value, indent=2, ensure_ascii=False)
    print(f"\n[DEBUG] {label}:\n{value}")


def tap(label: str, pick=None):
    """A pass-through Runnable that prints what flows through it.

    Put it anywhere in a chain with | and the data continues unchanged:
        chain = prompt | llm | tap("raw model output") | parser

    pick: optional function to select part of the value for printing.
    """

    def _tap(value):
        trace(label, pick(value) if pick else value)
        return value  # unchanged - this is why it is safe mid-chain

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
    """Raw JSON string -> validated dict. No Pydantic, as the assignment requires."""
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
# The escalation rule
# ----------------------------------------------------------------------
def needs_escalation(sentiment: dict) -> bool:
    """Assignment rule: Negative sentiment AND urgency of 4 or 5."""
    return sentiment["sentiment"] == "Negative" and sentiment["urgency"] >= 4


def add_escalation_tag(text: str) -> str:
    """Prepend the operational tag, without duplicating it."""
    text = text.strip()
    if text.startswith(ESCALATION_TAG):
        return text
    return f"{ESCALATION_TAG}\n\n{text}"


# ----------------------------------------------------------------------
# Chain builders
# ----------------------------------------------------------------------
def make_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    """system + human prompt (triage chains and the tone refiner)."""
    block = prompts[name]
    return ChatPromptTemplate.from_messages(
        [("system", block["system"]), ("human", block["human"])]
    )


def make_draft_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    """system + history + human prompt (the three department drafts)."""
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
    """Conditional routing (Task 3). Input needs: category, history, query."""
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
        general,  # default: General Inquiry and anything unexpected
    )


def build_refiner_chain(prompts: dict, llm):
    """Tone refiner (Task 4). Input needs: draft, sentiment, urgency, escalate."""
    return make_prompt(prompts, "tone_refiner") | llm | StrOutputParser()


def build_response_chain(prompts: dict, llm):
    """Sequential chaining: router -> refiner -> tag."""
    router = build_router_chain(prompts, llm)
    refiner = build_refiner_chain(prompts, llm)

    def refiner_inputs(x: dict) -> dict:
        """Flatten the sentiment dict into the fields the refiner prompt wants."""
        return {
            "draft": x["draft"],
            "sentiment": x["sentiment"]["sentiment"],
            "urgency": x["sentiment"]["urgency"],
            "escalate": "YES" if needs_escalation(x["sentiment"]) else "NO",
        }

    def finalize(x: dict) -> str:
        escalate = needs_escalation(x["sentiment"])
        trace("escalation rule (Negative AND urgency>=4)", escalate)
        if escalate:
            return add_escalation_tag(x["polished"])
        return x["polished"].strip()

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
# Main loop
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer Support Ticket Triage & Response Router"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print intermediate pipeline values to the terminal",
    )
    return parser.parse_args()


def main() -> None:
    global DEBUG
    DEBUG = parse_args().debug

    prompts = load_prompts()
    llm = build_llm()
    triage_chain = build_triage_chain(prompts, llm)
    response_chain = build_response_chain(prompts, llm)

    history: list = []

    print("Support Triage Console. Type 'exit' to quit.")
    print(f"Debug tracing: {'ON' if DEBUG else 'OFF (run with --debug to enable)'}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        # Step 2: classify and score sentiment in parallel.
        triage = triage_chain.invoke({"query": query})
        print(f"\n[Triage] Category : {triage['category']}")
        print(f"[Triage] Sentiment: {triage['sentiment']}")

        # Steps 3 + 4: route to a draft, then polish the tone.
        reply = response_chain.invoke(
            {
                "category": triage["category"],
                "sentiment": triage["sentiment"],
                "history": history,
                "query": query,
            }
        )
        print(f"\nAssistant: {reply}\n")

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=reply))
        trace("conversation history length (messages)", len(history))


if __name__ == "__main__":
    main()