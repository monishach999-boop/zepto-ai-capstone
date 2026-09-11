# Module 3 - Zepto AI Support Assistant

This module implements an offline GenAI customer support assistant for Zepto using Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.

The assistant answers questions related to Zepto policies using a local document corpus and deterministic mock responses without requiring an external LLM or API key.

## Document Corpus

The knowledge base contains 8 Zepto policy documents:

- doc_01 - Delivery Policy
- doc_02 - Returns & Refunds
- doc_03 - Membership Tiers
- doc_04 - Order Tracking
- doc_05 - Order Cancellation Policy
- doc_06 - Damaged or Missing Items
- doc_07 - Gift Cards
- doc_08 - Customer Support Hours

## Architecture

The workflow is:

Customer Query  
→ Intent Classification  
→ Conditional Routing  
→ ChromaDB Retrieval or Direct Answer  
→ Structured JSON Response

For policy questions, the query is embedded using `all-MiniLM-L6-v2` and the top 3 relevant documents are retrieved from ChromaDB.

For general questions, the assistant returns a fixed response explaining that it currently answers only Zepto policy questions.

## LangGraph Workflow

The StateGraph contains three main nodes:

1. `classify_intent` - Classifies the query as `policy_question` or `general_question`.
2. `retrieve_and_answer` - Retrieves the top 3 relevant policy documents and generates a deterministic answer.
3. `direct_answer` - Handles questions unrelated to Zepto policies.

Conditional routing determines which answer node is executed.

## Mock Mode

The project uses offline mock mode:

```python
MOCK_LLM = 1
```

No external LLM or API key is required for the graded baseline.

## Structured Output

The API response is validated using Pydantic.

Response format:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
```

## FastAPI

Run the API from the project root using:

```bash
uvicorn support_assistant.main:app
```

Open Swagger UI at:

`http://127.0.0.1:8000/docs`

Use the POST `/ask` endpoint to submit questions.

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

## Main Files

- `app.py` - Loads the 8 policy documents, creates embeddings, and stores them in ChromaDB.
- `graph.py` - Implements the LangGraph workflow and retrieval logic.
- `prompt_template.py` - Contains the structured prompt template and few-shot example.
- `schemas.py` - Defines the Pydantic response schema.
- `main.py` - Provides the FastAPI application and POST `/ask` endpoint.
- `docs/` - Contains the 8 Zepto policy documents.
- `chroma_db/` - Stores the persistent ChromaDB vector database.

## Technologies Used

- Python
- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Uvicorn

## Status

Module 3 AI Support Assistant successfully implements document retrieval, deterministic LangGraph routing, structured JSON output, and a working FastAPI endpoint in offline mock mode.