# Module 3 - LangGraph Support Workflow

from typing import TypedDict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from support_assistant.schemas import SupportResponse


# State used by LangGraph
class SupportState(TypedDict):
    query: str
    intent: str
    context: str
    answer: str
    sources: list[str]
    confidence: float


# Mock mode required for grading
MOCK_LLM = 1


# Policy keywords used for deterministic routing
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


# Paths
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# Load ChromaDB
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="zepto_support_docs"
)


# Load local embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Node 1 - Classify intent
def classify_intent(state: SupportState):
    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        return {
            "intent": "policy_question"
        }

    return {
        "intent": "general_question"
    }


# Router
def route_query(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve"

    return "direct"


# Node 2 - Retrieve top 3 documents and answer
def retrieve_and_answer(state: SupportState):

    query_embedding = model.encode(
        [state["query"]]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]
    source_ids = results["ids"][0]

    context = "\n".join(documents)

    # Deterministic MOCK_LLM response
    answer = (
        "Based on the retrieved context: "
        + documents[0][:200]
    )

    return {
        "context": context,
        "answer": answer,
        "sources": source_ids,
        "confidence": 1.0
    }


# Node 3 - Direct response for general questions
def direct_answer(state: SupportState):

    return {
        "answer": (
            "I can only answer questions "
            "about Zepto policies right now."
        ),
        "sources": [],
        "confidence": 1.0
    }


# Build LangGraph
builder = StateGraph(SupportState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)


# Start
builder.add_edge(
    START,
    "classify_intent"
)


# Conditional routing
builder.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "retrieve": "retrieve_and_answer",
        "direct": "direct_answer"
    }
)


# End
builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)


# Compile graph
graph = builder.compile()


# Test the workflow
if __name__ == "__main__":

    test_state: SupportState = {
        "query": "What is Zepto's refund policy?",
        "intent": "",
        "context": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }

    result = graph.invoke(test_state)

    # Validate using Pydantic
    validated = SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    print("Query:", result["query"])
    print("Intent:", result["intent"])

    print("\nValidated JSON Response:")

    print(
        validated.model_dump_json(
            indent=2
        )
    )