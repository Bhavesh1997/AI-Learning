"""
Customer Support Ticket Triage & Response Router
------------------------------------------------
STEP 1: terminal loop + conversation history (Task 1).
Later steps add: parallel triage, routing, tone refiner, --debug.
"""

import os
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Read OPENAI_API_KEY from the .env file in the AI TRANSITION root.
load_dotenv(find_dotenv())

# prompts.yaml sits next to this script, wherever you run it from.
PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")


def load_prompts() -> dict:
    """Read the YAML file and return it as a normal Python dict."""
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm() -> ChatOpenAI:
    """Create the chat model. Model name can be changed from .env."""
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def build_general_chain(prompts: dict, llm):
    """prompt -> model -> plain string. This is our first LCEL chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["general_draft"]["system"]),
            MessagesPlaceholder("history"),  # past messages get inserted here
            ("human", "{query}"),            # the new customer message
        ]
    )
    return prompt | llm | StrOutputParser()


def main() -> None:
    prompts = load_prompts()
    llm = build_llm()
    chain = build_general_chain(prompts, llm)

    history: list = []  # list of HumanMessage / AIMessage objects

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

        reply = chain.invoke({"history": history, "query": query})
        print(f"\nAssistant: {reply}\n")

        # Save this turn AFTER answering, so it is available next turn.
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=reply))


if __name__ == "__main__":
    main()
