"""Offline tests: the LLM is replaced by a fake, so no API key or network needed."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import pipeline
from app.main import app


class FakeLLM(GenericFakeChatModel):
    """Answers by looking at the system prompt, so parallel calls stay deterministic."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        if "ticket classifier" in system:
            text = "Billing"
        elif "emotional tone" in system:
            text = '{"sentiment": "Negative", "urgency": 5}'
        else:
            text = "Draft reply text."
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def make_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(pipeline, "build_llm", lambda: FakeLLM(messages=iter([])))
    return TestClient(app)


def test_health(monkeypatch):
    with make_client(monkeypatch) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_triage_escalates_and_keeps_session(monkeypatch):
    with make_client(monkeypatch) as c:
        r = c.post("/triage", json={"query": "I was charged twice!", "debug": True})
        assert r.status_code == 200
        body = r.json()
        assert body["category"] == "Billing"
        assert body["escalated"] is True
        assert body["reply"].startswith("[URGENT ESCALATION]")
        assert body["debug_trace"]

        r2 = c.post("/triage", json={"query": "any update?", "session_id": body["session_id"]})
        assert r2.json()["session_id"] == body["session_id"]
        assert r2.json()["debug_trace"] is None


def test_empty_query_rejected(monkeypatch):
    with make_client(monkeypatch) as c:
        assert c.post("/triage", json={"query": ""}).status_code == 422
