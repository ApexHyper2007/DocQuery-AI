# DocQuery AI

DocQuery AI is a local Retrieval-Augmented Generation (RAG) assistant for technical questions about official PyTorch, torchvision, and NVIDIA CUDA documentation. It searches documentation passages and asks a local Ollama model to answer with source labels. If retrieval evidence is weak, the service returns an insufficient-context response instead of asking the model to guess.

## Architecture

```text
Official docs -> HTML ingestion -> Markdown-aware chunks -> MiniLM embeddings -> ChromaDB
                                                                  |
Question -> BM25 keyword search + dense vector search -> reciprocal-rank fusion
                                                                  |
                                       Ollama grounded generation -> FastAPI -> Streamlit
```

The BM25 implementation is deliberately small and visible in `src/retrieval.py`; it does not require a separate search service. Chroma stores persistent 384-dimensional MiniLM vectors. Ollama is a separate local runtime (not a Python package) and the default model is `llama3.2:3b`; choose a smaller quantized model if your local memory budget requires it. Embeddings and generation default to CPU/local execution, and `DOCQUERY_DEVICE=cuda` is an explicit opt-in for embedding.

## Current status

Stages 1-10 are implemented: ingestion and redirect validation, HTML cleaning, structure-aware chunking, embedding/indexing, BM25 and dense retrieval, reciprocal-rank fusion, Ollama answer generation, citation validation and insufficient-context handling, FastAPI, Streamlit, unit tests, and source-level retrieval evaluation. The project is published on GitHub; use focused commits for future changes.

The current local corpus is built from six official documentation pages. The ignored local index contains 1,000 chunks. The evaluation examples are a small smoke benchmark, not a representative quality guarantee. Review retrievals and add labeled questions before tuning thresholds or claiming measured quality.

## Windows setup (PowerShell)

Use an isolated environment so these dependencies do not alter another Python or PyTorch setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The embedding model downloads on the first index/retrieval run and is cached under ignored `.cache/`. Create or refresh the corpus and vector index with:

```powershell
python .\src\ingest_docs.py --refresh
python .\src\process_docs.py
python .\src\index_docs.py
```

If the index already exists, skip those commands and start the services.

Install Ollama separately from its official distribution, then in PowerShell:

```powershell
ollama pull llama3.2:3b
ollama serve
```

Start the API in another PowerShell window:

```powershell
cd C:\path\to\DocQuery-AI
.\.venv\Scripts\Activate.ps1
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Start the UI in a third window:

```powershell
cd C:\path\to\DocQuery-AI
.\.venv\Scripts\Activate.ps1
streamlit run .\src\streamlit_app.py
```

Open the local Streamlit URL shown in the terminal. API documentation is at `http://127.0.0.1:8000/docs`.

## Configuration

Environment variables are optional and have local defaults. Set them in the current PowerShell session as needed:

```powershell
$env:DOCQUERY_OLLAMA_URL = "http://localhost:11434"
$env:DOCQUERY_OLLAMA_MODEL = "llama3.2:3b"
$env:DOCQUERY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
$env:DOCQUERY_DEVICE = "cpu"
$env:DOCQUERY_COLLECTION = "docquery_docs"
$env:DOCQUERY_RRF_K = "60"
$env:DOCQUERY_MIN_SIMILARITY = "0.20"
$env:DOCQUERY_MIN_BM25_SCORE = "1.5"
$env:DOCQUERY_BM25_K1 = "1.5"
$env:DOCQUERY_BM25_B = "0.75"
```

No API key is needed for the local Ollama setup. Do not put credentials in source files; `.env` is ignored by Git. The query endpoint accepts `question`, `top_k` (1-20), and `candidate_k` (1-100). Retrieval defaults to BM25 top 20 plus dense top 20, fused by reciprocal rank. Context thresholds are configurable because they need calibration against relevant and irrelevant questions for the corpus.

## Tests and retrieval evaluation

Run tests:

```powershell
python -m pytest -q
```

Run the labeled source-level evaluation after building the index and downloading the embedding model:

```powershell
python -m src.evaluate_retrieval --k 5
```

The evaluator reports hit rate, source-level precision/recall at k, and mean reciprocal rank (MRR). Its six examples check whether the expected source page appears, not whether a passage fully answers the query. For meaningful engineering evaluation, expand the set, label relevant chunk IDs, measure chunk-level Recall@k/MRR/NDCG, inspect failures, and compare changes using a fixed query set.

Quick API smoke checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/query `
  -ContentType "application/json" `
  -Body '{"question":"What does torch.Tensor.add do?"}'
```

## Repository contents

- `src/ingest_docs.py`: allowlisted downloads, redirect handling, validation, and manifest.
- `src/process_docs.py`: HTML-to-Markdown cleaning, heading-aware sections, code-aware chunking, and metadata.
- `src/index_docs.py`: MiniLM embeddings and persistent Chroma storage.
- `src/retrieval.py`: BM25, dense search, and reciprocal-rank fusion.
- `src/answering.py`, `src/ollama_client.py`: evidence threshold, grounded prompt, and local generation.
- `src/api.py`: FastAPI health and query endpoints.
- `src/streamlit_app.py`: chat UI with expandable source excerpts.
- `src/evaluate_retrieval.py`: retrieval metrics over `tests/eval_questions.json`.
- `data/raw`, `data/processed`, `data/index`: local generated artifacts; excluded from Git.

## Data, privacy, and Git

Raw downloads, processed chunks, the Chroma database, local virtual environment, model caches, and secrets are ignored by `.gitignore`. Query text and excerpts are sent to the local Ollama service on this machine; this code does not send them to a hosted model. Confirm local runtime configuration before indexing private sources.

Before each GitHub push, inspect `git status --short` and `git diff --cached --check`, confirm generated data and `.env` are absent, and review license/attribution terms for any documentation content you choose to redistribute. The repository is [ApexHyper2007/DocQuery-AI](https://github.com/ApexHyper2007/DocQuery-AI).

`.gitattributes` keeps source and documentation files on LF line endings in Git, including when developers work from Windows.
