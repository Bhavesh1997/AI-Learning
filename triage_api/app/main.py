"""FastAPI wrapper around the triage pipeline.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""

import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

from app.pipeline import TriagePipeline, start_trace, stop_trace  # noqa: E402

MAX_SESSIONS = 1000  # oldest sessions are dropped beyond this
MAX_HISTORY_MESSAGES = 40  # keeps prompts (and cost) bounded


# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class TriageRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Customer message")
    session_id: str | None = Field(
        None, description="Reuse the id from a previous response to keep conversation history"
    )
    debug: bool = Field(False, description="Include intermediate pipeline steps in the response")


class TriageResponse(BaseModel):
    session_id: str
    category: str
    sentiment: str
    urgency: int
    escalated: bool
    reply: str
    debug_trace: list[dict] | None = None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = TriagePipeline()  # fails fast if OPENAI_API_KEY is missing
    app.state.sessions = OrderedDict()  # session_id -> list of messages
    yield


app = FastAPI(
    title="Support Ticket Triage API",
    description="Classifies a support message, scores sentiment, routes to a "
    "department draft, polishes the tone and flags urgent escalations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
async def triage(req: TriageRequest) -> TriageResponse:
    sessions: OrderedDict = app.state.sessions
    session_id = req.session_id or uuid.uuid4().hex
    history = sessions.get(session_id, [])

    sink = start_trace() if req.debug else None
    try:
        result = await app.state.pipeline.run(req.query, history)
    except Exception as exc:  # LLM/network/auth errors
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {type(exc).__name__}") from exc
    finally:
        stop_trace()

    history = (history + [HumanMessage(content=req.query), AIMessage(content=result["reply"])])[
        -MAX_HISTORY_MESSAGES:
    ]
    sessions[session_id] = history
    sessions.move_to_end(session_id)
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)

    return TriageResponse(session_id=session_id, debug_trace=sink, **result)
