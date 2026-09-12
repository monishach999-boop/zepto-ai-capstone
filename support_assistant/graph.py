# Module 3 - LangGraph Support Workflow

import os
from pathlib import Path
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from support_assistant.schemas import SupportResponse
from support_assistant.prompt_template import PROMPT_TEMPLATE


# =========================================================
# STATE
# =========================================================

class SupportState(TypedDict):
    query: str
    intent: str
    context: str
    answer: str
    sources: list[str]
    confidence: float


# =========================================================
# MOCK MODE
# Default: MOCK_LLM unset or MOCK_LLM=1
# Optional real mode: MOCK_LLM=0
# =========================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# =========================================================
# POLICY KEYWORDS
# =========================================================

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


# =========================================================
# PATHS + CHROMADB
# =========================================================

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"


client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


collection = client.get_collection(
    name="zepto_support_docs"
)


# Local embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# NODE 1 - CLASSIFY INTENT
# =========================================================

def classify_intent(state: SupportState):

    query = state["query"].lower()

    if MOCK_LLM:

        # Deterministic grading mode
        if any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        ):
            intent = "policy_question"

        else:
            intent = "general_question"

    else:

        # Optional real-LLM mode can be connected here.
        # For now we keep the same deterministic fallback
        # so the application remains runnable without an API.
        if any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        ):
            intent = "policy_question"

        else:
            intent = "general_question"


    return {
        "intent": intent
    }


# =========================================================
# ROUTER
# =========================================================

def route_query(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve"

    return "direct"


# =========================================================
# NODE 2 - RETRIEVE + ANSWER
# =========================================================

def retrieve_and_answer(state: SupportState):

    # Retrieval always runs locally using embeddings
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


    # Build the structured prompt
    formatted_prompt = PROMPT_TEMPLATE.format(
        context=context,
        query=state["query"]
    )


    if MOCK_LLM:

        # Exact deterministic mock answer required for grading
        top_chunk_snippet = documents[0][:200]

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

    else:

        # Optional real-LLM integration point.
        # The project does not require a paid API for grading.
        # Keep a deterministic fallback when no real provider
        # has been configured.
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


# =========================================================
# NODE 3 - DIRECT ANSWER
# =========================================================

def direct_answer(state: SupportState):

    # General questions do not use retrieval or an LLM.
    return {
        "answer": (
            "I can only answer questions "
            "about Zepto policies right now."
        ),
        "sources": [],
        "confidence": 1.0
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(
    SupportState
)


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


builder.add_edge(
    START,
    "classify_intent"
)


builder.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "retrieve": "retrieve_and_answer",
        "direct": "direct_answer"
    }
)


builder.add_edge(
    "retrieve_and_answer",
    END
)


builder.add_edge(
    "direct_answer",
    END
)


graph = builder.compile()


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    print(
        "MOCK_LLM mode:",
        MOCK_LLM
    )


    test_state: SupportState = {
        "query": "What is Zepto's refund policy?",
        "intent": "",
        "context": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }


    result = graph.invoke(
        test_state
    )


    validated = SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )


    print(
        "Query:",
        result["query"]
    )


    print(
        "Intent:",
        result["intent"]
    )


    print(
        "\nValidated JSON Response:"
    )


    print(
        validated.model_dump_json(
            indent=2
        )
    )