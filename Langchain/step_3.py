"""
Customer Support Ticket Triage & Response Router
------------------------------------------------
STEP 1: terminal loop + conversation history         (Task 1)
STEP 2: parallel triage - classifier + sentiment     (Task 2)
STEP 3: conditional routing to department drafts     (Task 3)   <-- this file
Later steps add: tone refiner (Task 4), --debug (Task 5).
"""

import json
import os
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")

CATEGORIES = ["Technical Support", "Billing", "General Inquiry"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
DEFAULT_SENTIMENT = {"sentiment": "Neutral", "urgency": 3}


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
# Chain builders
# ----------------------------------------------------------------------
def make_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    """system + human prompt, for the triage chains."""
    block = prompts[name]
    return ChatPromptTemplate.from_messages(
        [("system", block["system"]), ("human", block["human"])]
    )


def make_draft_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    """system + history + human prompt, for the three department drafts."""
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
        | RunnableLambda(normalize_category)
    )


def build_sentiment_chain(prompts: dict, llm):
    return (
        make_prompt(prompts, "sentiment")
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_sentiment)
    )


def build_triage_chain(prompts: dict, llm):
    """Classifier and sentiment run in parallel on the same input.
    Output: {"category": "...", "sentiment": {...}}"""
    return RunnableParallel(
        category=build_classifier_chain(prompts, llm),
        sentiment=build_sentiment_chain(prompts, llm),
    )


def build_router_chain(prompts: dict, llm):
    """Conditional routing (Task 3).

    Input dict must contain: category, history, query.
    RunnableBranch checks each (condition, chain) pair in order and runs the
    FIRST chain whose condition returns True. The last argument is the default.
    """
    technical = make_draft_prompt(prompts, "technical_draft") | llm | StrOutputParser()
    billing = make_draft_prompt(prompts, "billing_draft") | llm | StrOutputParser()
    general = make_draft_prompt(prompts, "general_draft") | llm | StrOutputParser()

    return RunnableBranch(
        (lambda x: x["category"] == "Technical Support", technical),
        (lambda x: x["category"] == "Billing", billing),
        general,  # default branch: General Inquiry and anything unexpected
    )


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main() -> None:
    prompts = load_prompts()
    llm = build_llm()
    triage_chain = build_triage_chain(prompts, llm)
    router_chain = build_router_chain(prompts, llm)

    history: list = []

    print("Support Triage Console. Type 'exit' to quit.\n")
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
        category = triage["category"]
        sentiment = triage["sentiment"]

        print(f"\n[Triage] Category : {category}")
        print(f"[Triage] Sentiment: {sentiment}")

        # Step 3: route to the matching department draft.
        draft = router_chain.invoke(
            {"category": category, "history": history, "query": query}
        )
        print(f"\nAssistant: {draft}\n")

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=draft))


if __name__ == "__main__":
    main()