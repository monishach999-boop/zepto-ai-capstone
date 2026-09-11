# Module 3 - FastAPI Application

from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import graph
from support_assistant.schemas import SupportResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline LangGraph-based Zepto policy support API",
    version="1.0"
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Zepto Support Assistant API is running."}


@app.post("/ask", response_model=SupportResponse)
def ask_question(request: AskRequest):

    state = {
        "query": request.query,
        "intent": "",
        "context": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }

    result = graph.invoke(state)

    return SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )