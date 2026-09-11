# Zepto AI Capstone Project

This project is developed as part of the Zepto AI Capstone. It contains modules for data engineering, analytics, and an AI-based customer support assistant.

---

## Module 1 - Data Pipeline

The Data Pipeline module collects book product data from Books to Scrape and processes it into a clean and structured dataset.

### Data Source

Website used: Books to Scrape

The pipeline collects books from four categories:

- Travel
- Mystery
- Historical Fiction
- Sequential Art

The final dataset contains more than 60 books.

### Data Collected

For each book, the following information is collected:

- Title
- Price
- Rating
- Availability
- Category

### Data Cleaning

The scraped data is cleaned before storing:

- Book price is converted into a numeric GBP value.
- Star ratings such as One, Two, Three, Four, and Five are converted to integers from 1 to 5.
- Availability is converted into an `in_stock` boolean value.
- Missing or invalid numeric values are handled using median imputation.

### Currency Conversion

The project uses the fixed conversion rate required for the capstone:

**1 GBP = 105.50 INR**

The INR price is calculated as:

```text
price_inr = price_gbp × 105.50
```

This is a fixed project-defined constant and does not use a live exchange-rate API.

### Database

The cleaned data is stored in a SQLite database using two normalized tables:

- `categories`
- `books`

The tables are connected using a primary key and foreign key relationship.

### SQL and Pandas

The project performs SQL queries demonstrating:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

SQL query results are also loaded into pandas DataFrames.

The SQL JOIN result is compared with a pandas `merge()` operation to verify that both approaches produce the same result.

---

## Module 2 - Data Analytics and Machine Learning

The Analytics module performs exploratory data analysis and machine learning on the cleaned book dataset.

The module includes:

- Data exploration
- Statistical analysis
- Data visualization
- Feature preparation
- Machine learning modeling
- Model evaluation

The analytics code is available in the `analytics` directory.

---

## Module 3 - Zepto AI Support Assistant

Module 3 implements an offline AI-powered customer support assistant for Zepto.

It uses a local Zepto policy knowledge base together with Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.

### Policy Knowledge Base

The assistant uses 8 local documents covering:

- Delivery Policy
- Returns & Refunds
- Membership Tiers
- Order Tracking
- Order Cancellation Policy
- Damaged or Missing Items
- Gift Cards
- Customer Support Hours

### Embeddings and ChromaDB

The policy documents are embedded using:

```text
all-MiniLM-L6-v2
```

The embeddings and documents are stored in a persistent ChromaDB collection.

For a policy-related customer query, the system retrieves the top 3 most relevant documents.

### LangGraph Architecture

The workflow is:

```text
Customer Query
      ↓
Intent Classification
      ↓
Conditional Routing
   ↙       ↘
Policy    General
Question  Question
   ↓         ↓
Retrieve   Direct
& Answer   Answer
   ↘         ↙
Structured JSON Response
```

The LangGraph workflow contains three main nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

### Offline Mock Mode

The graded baseline operates in deterministic offline mock mode:

```python
MOCK_LLM = 1
```

No external LLM or API key is required.

Policy questions use local retrieval, while unrelated general questions receive a fixed response.

### Structured Output

The final response is validated using a Pydantic schema containing:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
```

### FastAPI

The support assistant is exposed through a FastAPI application.

Start the API from the project root using:

```bash
uvicorn support_assistant.main:app
```

Then open Swagger UI at:

```text
http://127.0.0.1:8000/docs
```

The main endpoint is:

```text
POST /ask
```

### Example - Policy Question

Request:

```json
{
  "query": "What is Zepto's refund policy?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect...",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_03"
  ],
  "confidence": 1.0
}
```

The ordering of retrieved sources may vary based on embedding similarity.

### Example - General Question

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

### Module 3 Files

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── app.py
├── graph.py
├── main.py
├── prompt_template.py
├── schemas.py
└── README.md
```

---

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Docker

A Dockerfile is provided at the project root.

Build the Docker image using:

```bash
docker build -t zepto-support .
```

Run the container using:

```bash
docker run -p 7860:7860 zepto-support
```

The container serves the FastAPI application on port `7860`.

---

## Technologies Used

- Python
- Pandas
- SQLite
- Scikit-learn
- Sentence Transformers
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Uvicorn
- Docker

---

## Project Status

- Module 1 - Data Pipeline: Completed
- Module 2 - Analytics and Machine Learning: Completed
- Module 3 - AI Support Assistant: Completed
- FastAPI `/ask` endpoint: Tested successfully
- Offline mock mode: Implemented
- Dockerfile: Created