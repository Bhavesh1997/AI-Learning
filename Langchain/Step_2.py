"""
Customer Support Ticket Triage & Response Router
------------------------------------------------
STEP 1: terminal loop + conversation history        (Task 1)
STEP 2: parallel triage - classifier + sentiment     (Task 2)   <-- this file
Later steps add: routing, tone refiner, --debug.
"""

import json
import os
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")

CATEGORIES = ["Technical Support", "Billing", "General Inquiry"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
DEFAULT_SENTIMENT = {"sentiment": "Neutral", "urgency": 3}  # safe fallback


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def load_prompts() -> dict:
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


# ----------------------------------------------------------------------
# Output cleaners (plain Python functions that fix up model text)
# ----------------------------------------------------------------------
def normalize_category(raw: str) -> str:
    """Map whatever the model said to one of our 3 exact labels."""
    text = raw.strip().lower()
    for category in CATEGORIES:
        if category.lower() in text:
            return category
    return "General Inquiry"  # unknown answer -> safest bucket


def parse_sentiment(raw: str) -> dict:
    """Turn the model's raw JSON string into a validated Python dict.
    No Pydantic: just json.loads plus manual checks (as the assignment says)."""
    text = raw.strip()

    # Models sometimes wrap JSON in ```json ... ``` even when told not to.
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
    urgency = max(1, min(5, urgency))  # force into the 1-5 range

    return {"sentiment": sentiment, "urgency": urgency}


# ----------------------------------------------------------------------
# Chains
# ----------------------------------------------------------------------
def make_prompt(prompts: dict, name: str) -> ChatPromptTemplate:
    """Build a system+human prompt from a block in prompts.yaml."""
    block = prompts[name]
    return ChatPromptTemplate.from_messages(
        [("system", block["system"]), ("human", block["human"])]
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
    """Run the classifier and sentiment chains IN PARALLEL on the same input.
    Output is a dict: {"category": "...", "sentiment": {...}}"""
    return RunnableParallel(
        category=build_classifier_chain(prompts, llm),
        sentiment=build_sentiment_chain(prompts, llm),
    )


def build_general_chain(prompts: dict, llm):
    """Step 1 chain. Still used as a placeholder reply until Step 3 adds routing."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["general_draft"]["system"]),
            MessagesPlaceholder("history"),
            ("human", "{query}"),
        ]
    )
    return prompt | llm | StrOutputParser()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main() -> None:
    prompts = load_prompts()
    llm = build_llm()
    triage_chain = build_triage_chain(prompts, llm)
    general_chain = build_general_chain(prompts, llm)

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

        # Step 2: both sub-chains run at the same time on the same query.
        triage = triage_chain.invoke({"query": query})
        print(f"\n[Triage] Category : {triage['category']}")
        print(f"[Triage] Sentiment: {triage['sentiment']}")

        # Placeholder reply (Step 3 will route by category instead).
        reply = general_chain.invoke({"history": history, "query": query})
        print(f"\nAssistant: {reply}\n")

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=reply))


if __name__ == "__main__":
    main()