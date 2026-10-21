"""
Customer Support Ticket Triage & Response Router
------------------------------------------------
STEP 1: terminal loop + conversation history         (Task 1)
STEP 2: parallel triage - classifier + sentiment     (Task 2)
STEP 3: conditional routing to department drafts     (Task 3)
STEP 4: sequential tone refiner + escalation rule    (Task 4)   <-- this file
Next step adds: --debug tracer (Task 5).
"""

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
# Step 4 helpers: the escalation rule
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

    return RunnableBranch(
        (lambda x: x["category"] == "Technical Support", technical),
        (lambda x: x["category"] == "Billing", billing),
        general,  # default: General Inquiry and anything unexpected
    )


def build_refiner_chain(prompts: dict, llm):
    """Tone refiner (Task 4). Input needs: draft, sentiment, urgency, escalate."""
    return make_prompt(prompts, "tone_refiner") | llm | StrOutputParser()


def build_response_chain(prompts: dict, llm):
    """Sequential chaining: router -> refiner -> tag.

    Each stage feeds the next:
      in : {category, sentiment, history, query}
      1. add "draft"      (router output)
      2. add "polished"   (refiner output, which reads "draft")
      3. apply the tag if escalation is on
    """
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
        if needs_escalation(x["sentiment"]):
            return add_escalation_tag(x["polished"])
        return x["polished"].strip()

    return (
        RunnablePassthrough.assign(draft=router)
        | RunnablePassthrough.assign(
            polished=RunnableLambda(refiner_inputs) | refiner
        )
        | RunnableLambda(finalize)
    )


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main() -> None:
    prompts = load_prompts()
    llm = build_llm()
    triage_chain = build_triage_chain(prompts, llm)
    response_chain = build_response_chain(prompts, llm)

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


if __name__ == "__main__":
    main()
