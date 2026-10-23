from fastapi import FastAPI
from pydantic import BaseModel

from rag import graph, AnswerResponse


# ============================================================
# TASK 5: FASTAPI APPLICATION
# ============================================================


# ------------------------------------------------------------
# Create FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto customer support assistant",
    version="1.0.0"
)


# ------------------------------------------------------------
# Request model
# ------------------------------------------------------------

class AskRequest(BaseModel):
    query: str


# ------------------------------------------------------------
# POST /ask endpoint
# ------------------------------------------------------------

@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):

    # Run LangGraph
    result = graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
    )

    # Validate final response using Task 4 Pydantic model
    response = AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    return response