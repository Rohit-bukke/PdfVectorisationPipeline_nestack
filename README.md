# PDF Vectorisation Pipeline

A production-ready PDF vectorisation and similarity retrieval service built with **LangChain**, **LangGraph**, **ChromaDB**, and **FastAPI**.

## Live Deployment
https://YOUR-REAL-DEPLOYMENT-URL

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository and navigate into it
cd project_ai

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 2. Ingest PDF Documents
Ingest the PDF document into the persistent Chroma vector store:
```bash
python ingest.py --file data/sample.pdf
```
Options available:
- `--chunk-size` (default: `1000`)
- `--chunk-overlap` (default: `150`)
- `--persist-dir` (default: `./vectorstore`)
- `--collection-name` (default: `pdf_documents`)
- `--model-name` (default: `sentence-transformers/all-MiniLM-L6-v2`)

### 3. Run the Retrieval Service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Run Tests & Generate `results.json`
```bash
python generate_results.py
```
This executes API input validation tests (empty query, whitespace query, out-of-range `top_k`) and runs sample queries against the vector store, writing the outputs to `results.json`.

---

## Architecture

```text
PDF
 ↓
PyPDFLoader
 ↓
RecursiveCharacterTextSplitter
 ↓
HuggingFaceEmbeddings
 ↓
Chroma
 ↓
FastAPI /query
 ↓
similarity_search_with_score()
```

### Pipeline Details

1. **Ingestion Pipeline (`ingest.py`)**:
   - Structured as an idempotent **LangGraph** sequential workflow:
     $$\text{START} \longrightarrow \text{load} \longrightarrow \text{split} \longrightarrow \text{embed} \longrightarrow \text{store} \longrightarrow \text{END}$$
   - **Load**: Extracts text and metadata page-by-page from the source PDF via `PyPDFLoader`.
   - **Split**: Segments text using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 150) while tracking 1-indexed `page_number` metadata.
   - **Embed**: Generates dense embeddings locally via `HuggingFaceEmbeddings` using `sentence-transformers/all-MiniLM-L6-v2`.
   - **Store**: Indexes and persists embeddings into `Chroma` with deterministic chunk IDs for idempotent execution.

2. **Retrieval API (`app.py`)**:
   - FastAPI microservice that queries the Chroma vector store directly using `similarity_search_with_score()`.
   - **Pure Retrieval**: Strictly no LLMs, conversational wrappers, or QA chains—returns raw chunk text, original page numbers, and similarity scores.
   - Validates incoming requests (`query` non-empty, `top_k` between 1 and 20).

---

## Project Structure

```text
├── data/
│   └── sample.pdf             # Input PDF document
├── vectorstore/               # ChromaDB persistent storage directory
├── app.py                     # FastAPI retrieval server
├── ingest.py                  # LangGraph-based ingestion pipeline
├── generate_results.py        # Validation tests and results generator
├── results.json               # Real query results and benchmark output
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment configuration
├── .env                       # Local environment variables
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Multi-service container orchestration
└── .dockerignore              # Files excluded from Docker context
```

---

## API Specification

### `POST /query`
Performs vector similarity search against the indexed PDF chunks.

#### Request Body
```json
{
  "query": "What is the difference between supervised and unsupervised learning?",
  "top_k": 3
}
```

#### Response Body
```json
[
  {
    "chunk_text": "REGEN AI PREP\nAI Developer Intern -- Deep Dive Q&A\nTopic 2...",
    "page_number": 5,
    "score": 0.6477
  }
]
```

### `GET /health`
Returns service health status.
```json
{
  "status": "healthy"
}
```

---

## Docker & Docker Compose

### Run API with Docker Compose
```bash
docker compose up -d
```
The API will be available at `http://localhost:8000`.

### Run Ingestion with Docker Compose
```bash
docker compose run --rm ingest
```

### Build and Run with Docker directly
```bash
docker build -t pdf-retrieval-api .
docker run -p 8000:8000 -v $(pwd)/vectorstore:/app/vectorstore pdf-retrieval-api
```
