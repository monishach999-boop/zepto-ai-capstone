# Module 3 - Zepto AI Support Assistant

This module implements an offline Zepto customer-support assistant using Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.

The graded baseline is fully deterministic and works without any external LLM, paid service, or API key.

---

## Document Corpus

The local knowledge base contains 8 Zepto policy documents:

- `doc_01` - Delivery Policy
- `doc_02` - Returns & Refunds
- `doc_03` - Membership Tiers
- `doc_04` - Order Tracking
- `doc_05` - Order Cancellation Policy
- `doc_06` - Damaged or Missing Items
- `doc_07` - Gift Cards
- `doc_08` - Customer Support Hours

---

## RAG Architecture

The complete policy-question pipeline is:

```text
8 Policy Documents
        ↓
Document Ingestion
        ↓
SentenceTransformer Embeddings
(all-MiniLM-L6-v2)
        ↓
ChromaDB Vector Store
        ↓
Customer Query
        ↓
classify_intent
        ↓
Conditional Routing
   ↙                 ↘
Policy Question     General Question
   ↓                     ↓
retrieve_and_answer   direct_answer
   ↓                     ↓
Query Embedding       Fixed Response
   ↓
Top-3 Chroma Retrieval
   ↓
Retrieved Context
   ↓
Structured Prompt
   ↓
Mock Generation
   ↓
Pydantic Validation
        ↓
FastAPI JSON Response
```

---

## Stage 1 - Document Ingestion

`app.py` loads all eight policy documents from the `docs/` directory.

Each policy document is treated as one document chunk and receives an ID from `doc_01` to `doc_08`.

This simple per-document chunking strategy is sufficient because the supplied policy documents are short and focused on one policy topic.

---

## Stage 2 - Embedding and Vector Storage

The local Sentence Transformer model `all-MiniLM-L6-v2` converts each policy document into an embedding vector.

The embeddings, document text, and document IDs are stored in the persistent ChromaDB collection `zepto_support_docs`.

The vector database is stored locally in `support_assistant/chroma_db/`.

No external embedding API is required.

---

## Stage 3 - LangGraph Intent Classification

The LangGraph `StateGraph` begins with the `classify_intent` node.

The query is converted to lowercase and classified as a `policy_question` when it contains any of these keywords:

- delivery
- return
- refund
- membership
- tracking
- cancel
- gift card
- support hours

Otherwise, it is classified as a `general_question`.

The graph then uses a conditional edge to select the correct next node.

---

## Stage 4 - Retrieval

Policy questions are routed to the `retrieve_and_answer` node.

The customer query is embedded locally using `all-MiniLM-L6-v2`.

ChromaDB performs similarity search and returns the top 3 most relevant policy documents.

Their document IDs are returned through the `sources` field.

Retrieval is always performed locally and does not depend on an external LLM.

---

## Stage 5 - Prompt and Generation

The retrieved documents are combined into context.

The `retrieve_and_answer` node formats the structured prompt stored in `prompt_template.py`.

The prompt contains:

- Role
- Retrieved context
- Customer question
- Task
- Output format
- Length restriction
- Negative constraint
- Few-shot example

In the default graded mock mode, no external LLM is called.

The deterministic response is:

`Based on the retrieved context: {top_chunk_snippet}`

where `top_chunk_snippet` is approximately the first 200 characters of the most similar retrieved document.

This makes the graded output deterministic and reproducible.

---

## General Question Route

Questions that are not classified as Zepto policy questions are routed to `direct_answer`.

This node performs no ChromaDB retrieval and no LLM call.

It returns exactly:

`I can only answer questions about Zepto policies right now.`

For this route, `sources` is an empty list and `confidence` is `1.0`.

---

## MOCK_LLM Behaviour

Mock mode is controlled using the `MOCK_LLM` environment variable.

The code uses:

```python
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
```

Therefore:

- `MOCK_LLM` unset → Mock mode
- `MOCK_LLM=1` → Mock mode
- `MOCK_LLM=0` → Optional real-LLM extension point (currently uses deterministic fallback)

### Default Graded Mode

When `MOCK_LLM` is unset or equal to `1`:

- Intent routing is deterministic.
- Embeddings are generated locally.
- ChromaDB retrieval is real and local.
- No external LLM or API is called.
- Generated answers are deterministic.
- Confidence is `1.0`.

### Optional Real-LLM Mode

`MOCK_LLM=0` represents an optional extension point for connecting a real LLM.

The current project does not configure a real or paid LLM provider. Therefore, when `MOCK_LLM=0`, the current implementation continues to use the deterministic fallback response.

The graded baseline remains completely offline and reproducible.

---

## LangGraph Workflow

The graph contains three main nodes:

### 1. `classify_intent`

Determines whether the request is a `policy_question` or `general_question`.

### 2. `retrieve_and_answer`

For policy questions:

1. Embeds the query.
2. Retrieves the top 3 ChromaDB documents.
3. Combines them into context.
4. Formats `PROMPT_TEMPLATE`.
5. Produces the deterministic mock answer.
6. Returns source document IDs and confidence.

This is the node responsible for retrieval and policy-answer generation.

### 3. `direct_answer`

Handles non-policy questions without retrieval or LLM generation.

---

## Structured Output

The final response is validated using the Pydantic `SupportResponse` model.

The response contains:

- `answer` - Generated support answer.
- `sources` - List of retrieved document IDs.
- `confidence` - Confidence value between `0.0` and `1.0`.

Example response:

```json
{
  "answer": "string",
  "sources": ["doc_02"],
  "confidence": 1.0
}
```

---

## FastAPI

The support assistant is exposed through a FastAPI application.

Run the API from the project root using:

```bash
uvicorn support_assistant.main:app
```

Open Swagger UI at:

`http://127.0.0.1:8000/docs`

The API provides the POST `/ask` endpoint.

Request format:

```json
{
  "query": "What is Zepto's refund policy?"
}
```

---

## Example 1 - Policy Question

Request:

```json
{
  "query": "What is Zepto's refund policy?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_03"
  ],
  "confidence": 1.0
}
```

The exact ordering of retrieved sources may vary depending on embedding similarity.

---

## Example 2 - General Question

Request:

```json
{
  "query": "Who is the Prime Minister of India?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Docker

The FastAPI support assistant can also run inside Docker.

Build the Docker image from the project root:

```bash
docker build -t zepto-support .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support
```

Open Swagger UI at:

`http://127.0.0.1:7860/docs`

The Docker container serves the POST `/ask` endpoint locally on port `7860`.

---

## Main Files

- `app.py` - Loads the 8 policy documents, creates embeddings, and stores them in ChromaDB.
- `graph.py` - Implements LangGraph routing, ChromaDB retrieval, prompt preparation, and answer generation.
- `prompt_template.py` - Contains the structured prompt with constraints and a few-shot example.
- `schemas.py` - Defines the Pydantic response schema.
- `main.py` - Provides the FastAPI application and POST `/ask` endpoint.
- `docs/` - Contains the 8 Zepto policy documents.
- `chroma_db/` - Stores the persistent local ChromaDB vector database.

---

## Technologies Used

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Uvicorn
- Docker

---

## Status

Module 3 successfully implements:

- 8-document local policy knowledge base
- Local Sentence Transformer embeddings
- Persistent ChromaDB vector storage
- Top-3 similarity retrieval
- LangGraph with 3 nodes
- Conditional intent routing
- Structured prompt with a negative constraint and few-shot example
- Deterministic offline mock generation
- Pydantic JSON validation
- FastAPI POST `/ask`
- Docker deployment

The graded baseline can run without a paid LLM API or external LLM service.