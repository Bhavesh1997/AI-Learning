"""
chains.py — the two LLM "brains" of the reflection agent.

Enhancements over the basic version:
  1. load_dotenv() runs HERE, before ChatOpenAI is created (fixes API-key-at-import issue)
  2. The critic returns STRUCTURED output (a Pydantic object with a numeric score),
     not free text — so the graph can make decisions based on the score.
  3. Different temperatures: creative generator, strict/consistent critic.
"""

from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()  # must run before ChatOpenAI(...) reads OPENAI_API_KEY


# ---------------------------------------------------------------------------
# Structured output schema for the critic
# ---------------------------------------------------------------------------
class Critique(BaseModel):
    """Structured feedback on a tweet."""

    score: int = Field(
        ge=1, le=10,
        description="Overall quality from 1 (poor) to 10 (ready to post as-is). Be strict.",
    )
    strengths: List[str] = Field(description="What already works well in the tweet.")
    weaknesses: List[str] = Field(description="What hurts the tweet's quality or reach.")
    suggestions: str = Field(
        description="Concrete, actionable recommendations for the next revision "
        "(length, hook, virality, style, hashtags, etc.)."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
            " Output ONLY the tweet text.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet."
            " Score it from 1 to 10, list strengths and weaknesses, and give detailed"
            " recommendations, including requests for length, virality, style, etc."
            " Reserve 9-10 for tweets that are genuinely ready to post.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------------------------------------------------------------
# LLMs and chains
# ---------------------------------------------------------------------------
generator_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  # creative
critic_llm = ChatOpenAI(model="gpt-4o", temperature=0)       # consistent grading

generation_chain = generation_prompt | generator_llm
# with_structured_output -> the chain returns a Critique object, not an AIMessage
reflection_chain = reflection_prompt | critic_llm.with_structured_output(Critique)