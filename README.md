# Zepto AI Capstone Project

This project is developed as part of the Zepto AI Capstone. It contains three connected modules for data engineering, data analytics and machine learning, and an AI-based customer support assistant.

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

The Analytics module performs exploratory data analysis and machine learning using the Titanic dataset.

The module includes:

- Data profiling and cleaning
- Missing-value analysis
- Statistical analysis
- Outlier analysis
- Data visualization
- Feature preprocessing
- Classification modeling
- Imbalance handling
- Hyperparameter tuning
- Regression analysis
- Model evaluation

The analytics workflow is implemented using:

- `analytics/01_eda.py`
- `analytics/02_modeling.py`

A committed `titanic.csv` file provides the offline dataset used by the analytics pipeline.

The final fitted machine-learning pipeline is saved as a Joblib artifact so that preprocessing and prediction can be reused together.

---

## Module 3 - Zepto AI Support Assistant

Module 3 implements an offline GenAI customer support assistant for Zepto.

It uses a local Zepto policy knowledge base together with Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.

### Policy Knowledge Base

The assistant uses 8 local policy documents:

- `doc_01` - Delivery Policy
- `doc_02` - Returns & Refunds
- `doc_03` - Membership Tiers
- `doc_04` - Order Tracking
- `doc_05` - Order Cancellation Policy
- `doc_06` - Damaged or Missing Items
- `doc_07` - Gift Cards
- `doc_08` - Customer Support Hours

### Embeddings and ChromaDB

The policy documents are embedded locally using the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The document embeddings are stored in a persistent ChromaDB collection.

For policy-related customer queries, the system embeds the incoming query and retrieves the top 3 most similar documents from ChromaDB.

### Structured Prompt

The project includes a structured prompt template containing:

- Role
- Context
- Task
- Output format
- Length instruction
- Explicit negative constraint
- Few-shot example

The prompt instructs the assistant not to answer using information that is not present in the provided context.

### LangGraph Architecture

The support workflow is:

```text
Customer Query
      |
      v
Intent Classification
      |
      v
Conditional Routing
   /           \
Policy        General
Question      Question
   |             |
   v             v
Retrieve       Direct
& Answer       Answer
   \             /
    \           /
     v         v
Structured JSON Response
```

The LangGraph workflow contains three main nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

Policy-related queries are routed to the retrieval workflow, while unrelated general queries are routed to the direct-answer workflow.

### Offline Mock Mode

The graded baseline operates in deterministic offline mock mode:

```python
MOCK_LLM = 1
```

No external LLM or LLM API key is required.

The mock classifier identifies policy questions using the required policy keywords. Policy questions use local ChromaDB retrieval, while unrelated general questions receive a deterministic fixed response.

### Structured Output

The final response is validated using a Pydantic schema containing:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
```

The `confidence` value is constrained between 0 and 1.

For general questions, the `sources` list is empty.

### FastAPI

The support assistant is exposed through a FastAPI application.

The main endpoint is:

```text
POST /ask
```

The request format is:

```json
{
  "query": "customer question"
}
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

## Project Structure

The capstone is organized into three connected modules:

```text
Zepto_AI_Capstone/
├── data_pipeline/
├── analytics/
├── support_assistant/
├── Dockerfile
├── README.md
└── requirements.txt
```

- `data_pipeline/` - Collects, cleans, converts, and stores book data in SQLite.
- `analytics/` - Performs exploratory data analysis and machine-learning modeling using the Titanic dataset.
- `support_assistant/` - Implements the Zepto policy support assistant.

---

## Installation

Create and activate a Python virtual environment.

Install the required packages from the project root:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Data Pipeline

Run the data pipeline from the project root:

```bash
python data_pipeline/pipeline.py
```

### Analytics

Run the exploratory data analysis:

```bash
python analytics/01_eda.py
```

Run the machine-learning pipeline:

```bash
python analytics/02_modeling.py
```

### Support Assistant

First, initialize the local policy document embeddings and ChromaDB collection:

```bash
python support_assistant/app.py
```

Then start the FastAPI application:

```bash
uvicorn support_assistant.main:app
```

After the server starts, open the FastAPI Swagger UI in the browser at:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /ask` endpoint to test the Zepto Support Assistant.

---

## Docker

A `Dockerfile` is provided at the project root.

Build the Docker image using:

```bash
docker build -t zepto-support .
```

Run the container using:

```bash
docker run -p 7860:7860 zepto-support
```

The container is configured to serve the FastAPI application on port `7860`.

---

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- Scikit-learn
- Joblib
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
- FastAPI `POST /ask` endpoint: Tested successfully
- Offline mock mode: Implemented
- Structured Pydantic response: Implemented
- ChromaDB retrieval: Implemented
- Dockerfile: Created