"""HTTP interface of the MLOps agent.

  uvicorn agent.main:app --port 8002
  POST /ask {"question": "..."}        -> AgentResponse (answer, known facts, uncertainties, tool evidence)
  GET  /pending                        -> state-changing actions waiting for approval
  POST /confirm/{id}  /reject/{id}     -> the human decision; the model can never call these
  GET  /tools   GET /health
If AGENT_API_TOKEN is set, every route except /health requires the header X-Agent-Token.
"""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Path
from pydantic import BaseModel, Field

from agent.llm import GeminiLLM
from agent.logging_utils import configure
from agent.orchestrator import Orchestrator
from agent.schemas import AgentResponse, PendingAction, ToolResult

PENDING_ID = Path(..., pattern=r"^[0-9a-f]{8}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    configure(os.getenv("LOG_LEVEL", "INFO"))
    yield


app = FastAPI(title="MLOps AI Agent", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator(GeminiLLM())


def require_token(x_agent_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("AGENT_API_TOKEN")
    if expected and not (x_agent_token and hmac.compare_digest(x_agent_token, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-Agent-Token")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools", dependencies=[Depends(require_token)])
def tools(orch: Orchestrator = Depends(get_orchestrator)):
    return [{"name": t.name, "mutating": t.mutating, "description": t.description}
            for t in sorted(orch.tools.values(), key=lambda s: s.name)]


@app.post("/ask", response_model=AgentResponse, dependencies=[Depends(require_token)])
def ask(req: AskRequest, orch: Orchestrator = Depends(get_orchestrator)):
    return orch.run(req.question)


@app.get("/pending", response_model=list[PendingAction], dependencies=[Depends(require_token)])
def pending(orch: Orchestrator = Depends(get_orchestrator)):
    return orch.pending()


@app.post("/confirm/{pending_id}", response_model=ToolResult, dependencies=[Depends(require_token)])
def confirm(pending_id: str = PENDING_ID, orch: Orchestrator = Depends(get_orchestrator)):
    result = orch.confirm(pending_id)
    if result.tool == "confirm" and result.status == "rejected":
        raise HTTPException(status_code=404, detail=result.error)
    return result


@app.post("/reject/{pending_id}", dependencies=[Depends(require_token)])
def reject(pending_id: str = PENDING_ID, orch: Orchestrator = Depends(get_orchestrator)):
    if not orch.reject(pending_id):
        raise HTTPException(status_code=404, detail="unknown or expired confirmation id")
    return {"rejected": True}