# Module 3 - LangGraph Support Workflow

import os
import time
from pathlib import Path
from typing import TypedDict

import chromadb
import numpy as np
import requests
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
# MOCK / OPTIONAL REAL LLM MODE
# =========================================================

# Default graded mode:
# MOCK_LLM unset or MOCK_LLM=1 -> deterministic mock mode
# MOCK_LLM=0 -> optional real-LLM extension
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

# Optional Groq configuration for MOCK_LLM=0
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


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
# HELPER: DETERMINISTIC MOCK CLASSIFICATION
# =========================================================

def mock_classify(query: str) -> str:

    query_lower = query.lower()

    if any(
        keyword in query_lower
        for keyword in POLICY_KEYWORDS
    ):
        return "policy_question"

    return "general_question"


# =========================================================
# HELPER: OPTIONAL REAL LLM CALL WITH RETRY
# =========================================================

def call_real_llm(
    prompt: str,
    max_retries: int = 2
):

    # Real LLM mode is optional and ungraded.
    # If credentials are absent, return None so the
    # deterministic fallback can still keep the app usable.
    if not GROQ_API_KEY or not GROQ_MODEL:

        print(
            "Real-LLM configuration not found. "
            "Using deterministic fallback."
        )

        return None


    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }


    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }


    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            return (
                data["choices"][0]
                ["message"]
                ["content"]
                .strip()
            )


        except (
            requests.RequestException,
            KeyError,
            IndexError,
            TypeError,
            ValueError
        ) as error:

            print(
                f"Real-LLM attempt {attempt} failed:",
                error
            )

            if attempt < max_retries:

                # Small retry delay
                time.sleep(attempt)


    print(
        "Real-LLM retries exhausted. "
        "Using deterministic fallback."
    )

    return None


# =========================================================
# HELPER: TOP-3 COSINE RETRIEVAL FROM CHROMADB
# =========================================================

def retrieve_top_3_cosine(
    query_embedding
):

    # Read locally stored documents and embeddings
    stored = collection.get(
        include=[
            "documents",
            "embeddings"
        ]
    )


    ids = stored["ids"]
    documents = stored["documents"]

    stored_embeddings = np.array(
        stored["embeddings"],
        dtype=float
    )

    query_vector = np.array(
        query_embedding,
        dtype=float
    )


    # Cosine similarity:
    # dot(a,b) / (||a|| * ||b||)
    query_norm = np.linalg.norm(
        query_vector
    )

    document_norms = np.linalg.norm(
        stored_embeddings,
        axis=1
    )


    similarities = (
        stored_embeddings
        @ query_vector
    ) / (
        document_norms
        * query_norm
        + 1e-12
    )


    # Highest cosine similarities first
    top_indices = np.argsort(
        similarities
    )[::-1][:3]


    top_documents = [
        documents[index]
        for index in top_indices
    ]

    top_ids = [
        ids[index]
        for index in top_indices
    ]


    return (
        top_documents,
        top_ids
    )


# =========================================================
# NODE 1 - CLASSIFY INTENT
# =========================================================

def classify_intent(
    state: SupportState
):

    query = state["query"]


    if MOCK_LLM:

        # Required deterministic graded baseline
        intent = mock_classify(
            query
        )


    else:

        # Optional real-LLM classification
        classification_prompt = f"""
Classify the following customer query into exactly one label:

policy_question
general_question

Use policy_question when the query concerns Zepto policies such as
delivery, returns, refunds, membership, tracking, cancellation,
gift cards, or support hours.

Return only one label and nothing else.

Query:
{query}
"""


        llm_result = call_real_llm(
            classification_prompt
        )


        if llm_result:

            cleaned_result = (
                llm_result
                .strip()
                .lower()
            )

            if (
                cleaned_result
                == "policy_question"
            ):

                intent = (
                    "policy_question"
                )

            elif (
                cleaned_result
                == "general_question"
            ):

                intent = (
                    "general_question"
                )

            else:

                # Safe fallback when LLM returns
                # an unexpected format
                intent = mock_classify(
                    query
                )

        else:

            # Retry failure / missing configuration
            # falls back deterministically
            intent = mock_classify(
                query
            )


    return {
        "intent": intent
    }


# =========================================================
# ROUTER
# =========================================================

def route_query(
    state: SupportState
):

    # Routing depends only on the intent stored in state.
    # It does NOT directly depend on MOCK_LLM.

    if (
        state["intent"]
        == "policy_question"
    ):
        return "retrieve"

    return "direct"


# =========================================================
# NODE 2 - RETRIEVE + ANSWER
# =========================================================

def retrieve_and_answer(
    state: SupportState
):

    # Retrieval always runs locally in both modes.
    query_embedding = (
        model.encode(
            state["query"]
        )
        .tolist()
    )


    # Explicit cosine-similarity retrieval
    documents, source_ids = (
        retrieve_top_3_cosine(
            query_embedding
        )
    )


    context = "\n\n".join(
        documents
    )


    # Structured prompt
    formatted_prompt = (
        PROMPT_TEMPLATE.format(
            context=context,
            query=state["query"]
        )
    )


    if MOCK_LLM:

        # Exact deterministic mock answer
        # required by the assignment
        top_chunk_snippet = (
            documents[0][:200]
        )

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )


    else:

        # Optional real LLM generation
        real_answer = call_real_llm(
            formatted_prompt
        )


        if real_answer:

            answer = real_answer

        else:

            # Retry failure / missing optional
            # configuration -> deterministic fallback
            top_chunk_snippet = (
                documents[0][:200]
            )

            answer = (
                "Based on the retrieved context: "
                + top_chunk_snippet
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

def direct_answer(
    state: SupportState
):

    # General questions require no retrieval
    # and no LLM call in the graded baseline.
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
        "retrieve":
            "retrieve_and_answer",

        "direct":
            "direct_answer"
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


    # -----------------------------------------------------
    # POLICY QUESTION TEST
    # -----------------------------------------------------

    policy_state: SupportState = {
        "query":
            "What is Zepto's refund policy?",

        "intent": "",
        "context": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }


    policy_result = graph.invoke(
        policy_state
    )


    policy_validated = (
        SupportResponse(
            answer=
                policy_result["answer"],

            sources=
                policy_result["sources"],

            confidence=
                policy_result["confidence"]
        )
    )


    print(
        "\nPolicy Query:",
        policy_result["query"]
    )

    print(
        "Intent:",
        policy_result["intent"]
    )

    print(
        "\nValidated Policy JSON Response:"
    )

    print(
        policy_validated
        .model_dump_json(
            indent=2
        )
    )


    # -----------------------------------------------------
    # GENERAL QUESTION TEST
    # -----------------------------------------------------

    general_state: SupportState = {
        "query":
            "Who is the Prime Minister of India?",

        "intent": "",
        "context": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }


    general_result = graph.invoke(
        general_state
    )


    general_validated = (
        SupportResponse(
            answer=
                general_result["answer"],

            sources=
                general_result["sources"],

            confidence=
                general_result[
                    "confidence"
                ]
        )
    )


    print(
        "\nGeneral Query:",
        general_result["query"]
    )

    print(
        "Intent:",
        general_result["intent"]
    )

    print(
        "\nValidated General JSON Response:"
    )

    print(
        general_validated
        .model_dump_json(
            indent=2
        )
    )