# DBU AI Policy Chatbot

A RAG-based (Retrieval-Augmented Generation) chatbot for answering questions about Dallas Baptist University's AI policy and related global AI governance frameworks. It uses a local LLM via Ollama, ChromaDB as a vector store, PostgreSQL for metrics logging, and a FastAPI backend with a simple web frontend.

---

## How It Works

1. **Document Ingestion** — Policy documents (PDFs and Markdown) are chunked, embedded using Sentence Transformers, and stored in a local ChromaDB vector store.
2. **Query Pipeline** — User questions are classified (in-scope, out-of-scope, harmful, coding). In-scope questions trigger a two-tier retrieval: DBU policy first, then global frameworks as fallback.
3. **Answer Generation** — Retrieved chunks are passed to `llama3.2` running locally via Ollama to generate a grounded answer.
4. **Evaluation** — Every interaction is scored with the RAG Triad (faithfulness, answer relevance, context precision) and logged to PostgreSQL.

---

## Prerequisites

- [Python 3.13+](https://www.python.org/)
- [UV](https://docs.astral.sh/uv/) — package manager
- [Ollama](https://ollama.com/) — local LLM runtime
- [PostgreSQL](https://www.postgresql.org/) — for metrics logging

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/amcpengineer/ai-policy-chatbot.git
cd ai-policy-chatbot
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Update `DATABASE_URL` in `.env` with your PostgreSQL connection string.

### 4. Add the policy documents

> **The documents are not included in this repository.** You must add them manually before running the ingestion step.

Place the following files into the `app/docs/` directory:

| File | Description |
|------|-------------|
| `DBU_AI_Policy_KnowledgeBase.md` | DBU internal AI policy knowledge base |
| `DBU_Policy.pdf` | DBU AI policy document |
| `eu_ai_act.pdf` | EU AI Act |
| `nist_ai_rmf.pdf` | NIST AI Risk Management Framework |
| `unesco_ai_ethics.pdf` | UNESCO AI Ethics recommendation |
| `whitehouse_eo_ai.pdf` | White House Executive Order on AI |

### 5. Pull the LLM model

```bash
ollama pull llama3.2
```

### 6. Initialize the database

Make sure PostgreSQL is running and the database specified in your `.env` exists, then start the app once to auto-create the tables:

```bash
uv run uvicorn app.main:app
```

### 7. Ingest the documents

Run the ingestion pipeline to chunk, embed, and store all documents in ChromaDB:

```bash
uv run python -m app.intelligence.ingest
```

**Ingestion options:**

| Flag | Description |
|------|-------------|
| `--reset` | Wipe the vector store before ingesting |
| `--collection NAME` | Target a specific ChromaDB collection |
| `--file PATH` | Ingest a single file instead of the full `app/docs/` folder |

Example — reset and re-ingest everything:

```bash
uv run python -m app.intelligence.ingest --reset
```

---

## Running the App

```bash
uv run uvicorn app.main:app --reload
```

Open your browser at [http://localhost:8000](http://localhost:8000).

---

## Project Structure

```
app/
├── api/routes/        # FastAPI route definitions
├── core/              # Config and database engine
├── docs/              # Policy documents (not tracked in git — add manually)
├── intelligence/      # Ingest pipeline, RAG service, RAG Triad scorer, vector store
├── services/          # RAG orchestration service
├── static/            # Frontend (index.html)
├── models.py          # SQLAlchemy models
└── main.py            # FastAPI app entry point
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI |
| Vector Store | ChromaDB |
| Embeddings | Sentence Transformers |
| LLM | Ollama — llama3.2 |
| Database | PostgreSQL + SQLAlchemy |
| Package Manager | UV |
