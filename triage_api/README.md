# Support Ticket Triage API

FastAPI wrapper around the LangChain triage pipeline (`Langchain/Step_5.py`).

## Run locally
```powershell
cd triage_api
..\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your real OPENAI_API_KEY in .env
uvicorn app.main:app --reload
```
Interactive docs: http://localhost:8000/docs

## Endpoints
- `GET /health`
- `POST /triage` — body: `{"query": "...", "session_id": null, "debug": false}`
  Returns `session_id, category, sentiment, urgency, escalated, reply, debug_trace`.
  Send the returned `session_id` back to continue the same conversation.

## Tests (no API key needed)
```powershell
pytest -q
```

## Deploy
```powershell
docker build -t triage-api .
docker run -p 8000:8000 --env-file .env triage-api
```
Works on any container host (Render, Railway, Cloud Run, Azure, ...). Set
`OPENAI_API_KEY` as a secret/env var there; the container honours `$PORT`.

Note: conversation history is held in memory, so run a single instance
(or swap `app.state.sessions` for Redis) before scaling out.
