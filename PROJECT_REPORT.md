# DocQuery AI — implementation and ML engineering report

## Outcome and scope

DocQuery now has a local documentation-to-answer path: it downloads and validates six official documentation pages, cleans and chunks them, stores MiniLM embeddings in persistent Chroma, combines BM25 and dense retrieval, asks Ollama for a context-grounded answer, and exposes that workflow through FastAPI and Streamlit. The source tree includes tests and a small labeled retrieval benchmark. Nothing was pushed to GitHub.

The existing local corpus/index from the earlier indexing stage remains in place. Its recorded index metadata reports 1,000 chunks and 384-dimensional embeddings. Generated documents, vector files, model cache, and the virtual environment remain ignored by Git.

## Stage-by-stage design

### 1. Repository, ingestion, and source manifest

`src/ingest_docs.py` downloads an explicit allowlist of PyTorch, torchvision, and CUDA pages. Standard HTTP redirects are followed by the HTTP client. If a successful HTTP response is actually a JavaScript redirect, meta refresh, or redirect page with canonical/anchor fallback, the destination is resolved relative to the response's final URL. The script rejects stubs, unsafe hosts, loops, oversized/non-HTML responses, and records success/failure details in `data/raw/manifest.json`.

**ML engineer concepts:** data provenance, source allowlists, idempotent ingestion, content validation, checksums, and keeping original versus canonical/final URLs. Bad source data contaminates every downstream chunk and answer, so ingestion validation is part of model quality.

### 2. HTML cleaning and structure-aware chunking

`src/process_docs.py` removes navigation and page furniture, renders useful HTML into Markdown, and retains headings, links, lists, tables, and code fences. It splits at Markdown headings outside code fences, carries the heading path into metadata, and then uses LangChain's recursive character splitter with custom code-aware packing. The result is `data/processed/chunks.jsonl`.

**ML engineer concepts:** chunk size trades retrieval precision against context completeness; overlap protects facts split at boundaries but increases storage and duplicate results. Headings and code structure are valuable retrieval features for API questions. Metadata should travel with every chunk so answers can cite sources later.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450, chunk_overlap=60,
    separators=["\n\n", "\n", " "],
)
```

This is the general splitter; the implementation protects Markdown code fences and repeats section headings around chunks.

### 3. Embeddings and persistent vector storage

`src/index_docs.py` loads `sentence-transformers/all-MiniLM-L6-v2`, validates each chunk against the model token limit, encodes batches, normalizes vectors, and upserts them into the `docquery_docs` Chroma collection using cosine distance. It stores chunk text and citation metadata with the vector, and writes a local `index_info.json` containing dimensions, model, corpus hash, and counts. `--rebuild` resets only the named DocQuery collection.

**ML engineer concepts:** embeddings map semantic similarity to vector geometry; normalization and distance metric must agree; the model used for queries must match the indexed model; token count differs from character count. The model's 256-token input limit led to keeping chunks small enough to avoid silent truncation.

```python
vectors = model.encode(
    texts, normalize_embeddings=True, convert_to_numpy=True,
)
collection.upsert(ids=ids, documents=texts, metadatas=metadata, embeddings=vectors.tolist())
```

### 4. Keyword search with BM25

`src/retrieval.py` includes an educational Okapi BM25 implementation that tokenizes each chunk, counts term frequency, computes inverse document frequency, and scores query terms. It preserves common API identifiers such as `torch.Tensor.add` as tokens. BM25 is good at exact symbols, parameter names, and rare terms that semantic embeddings can blur.

**ML engineer concepts:** BM25 is a sparse lexical ranker, not an embedding model. `k1` controls term-frequency saturation and `b` controls document-length normalization. It is configured via `DOCQUERY_BM25_K1` and `DOCQUERY_BM25_B`.

```python
idf = log(1 + (N - df + 0.5) / (df + 0.5))
score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
```

### 5. Dense search and hybrid fusion

The same query is embedded by MiniLM and sent to Chroma. Dense retrieval finds semantically related passages; BM25 finds lexical matches. `reciprocal_rank_fusion` (RRF) merges their ranked lists by stable chunk ID. RRF uses ranks instead of directly summing BM25 scores and cosine distances, which have unrelated scales. The top candidate count and RRF constant are configurable.

**ML engineer concepts:** retrieve more candidates than will be shown to the generator, deduplicate overlapping hits, and fuse ranks carefully. RRF score is a ranking score, not a probability or calibrated relevance score. Use a labeled query set to tune `candidate_k`, `top_k`, and `rrf_k`.

```python
rrf_score[chunk_id] += 1.0 / (rrf_k + rank)
```

### 6. Local LLM integration

`src/ollama_client.py` calls Ollama's local `/api/chat` endpoint using Python's standard library. It sends only the retrieved excerpts and asks the configured model (default `llama3.2:3b`) to answer from those excerpts and cite `[S1]`, `[S2]`, etc. No extra Python Ollama client package is required. The model itself is managed separately by Ollama.

**ML engineer concepts:** the prompt sets evidence constraints; temperature is low for factual answers; context-window budget constrains `top_k`; a small local model may still make mistakes. Citation formatting is instructed by the prompt and UI sources are attached independently, but model-generated prose must still be reviewed for factual support.

### 7. Grounding, citations, and insufficient context

`src/answering.py` retrieves ranked passages, compares the strongest cosine similarity and BM25 score with configurable thresholds, and returns an insufficient-context message without calling the model if neither indicates adequate evidence. It labels and returns source metadata with excerpts and URLs.

**ML engineer concepts:** an LLM cannot recover facts absent from its context; abstention reduces unsupported answers. Thresholds are heuristics, not calibrated confidence. Evaluate false accepts and false abstentions on a labeled set before adjusting `DOCQUERY_MIN_SIMILARITY` and `DOCQUERY_MIN_BM25_SCORE`.

### 8. FastAPI backend

`src/api.py` exposes `GET /health` and `POST /api/query`. Request fields have bounds, the retrieval object is initialized lazily, and runtime failures return useful HTTP errors. OpenAPI docs are available at `/docs` while the local service is running.

**ML engineer concepts:** keep retrieval/generation code outside route definitions, validate inputs at the boundary, load expensive models once rather than per request, and separate liveness from query readiness.

### 9. Streamlit interface

`src/streamlit_app.py` provides a chat flow that calls the API and shows expandable source excerpts with section names and links. API and UI are separate processes so each can be debugged independently.

**ML engineer concepts:** make provenance visible in the interface; users need to inspect the evidence behind generated text. This first version is a local single-user UI and does not implement authentication or concurrent-session persistence.

### 10. Tests and retrieval evaluation

`tests/test_ingest_docs.py` covers redirect resolution and stub rejection. `tests/test_retrieval.py` covers API tokenization, BM25 ranking, RRF agreement, citation metadata, and weak-context refusal. `tests/test_api.py` covers health and query response shape without requiring Ollama. `tests/eval_questions.json` provides six manually labeled source-level questions, and `src/evaluate_retrieval.py` reports hit rate at k and MRR.

**ML engineer concepts:** unit tests verify code invariants, while retrieval evaluation checks ranking on labeled queries. Source-level labels are only a smoke metric; a stronger benchmark labels relevant chunk IDs and tracks Recall@k, MRR, and NDCG, plus human judgments of answer faithfulness and completeness. Keep an unchanged benchmark when tuning retrieval.

### 11. Performance and repository readiness

The defaults favor a 6 GB laptop GPU and reproducibility: embedding runs on CPU unless explicitly configured, vectors are batched, Chroma persists locally, and Ollama serves a small local model. `requirements.txt` now contains direct dependencies for each implemented layer. `.gitignore` excludes environments, secrets, model downloads, raw/processed corpora, and generated vector stores. `.env.example` contains non-secret defaults only. README documents commands, configuration, and evaluation.

**ML engineer concepts:** profile before optimizing; measure embedding latency, retrieval latency, generation latency, memory use, and quality together. Increasing batch sizes or using GPU can improve speed but changes resource use. For a portfolio release, add robust retrieval labels, screenshots or a demo recording, a license review for redistributed source material, and a clean-install verification.

## Key files changed/added in this completion

- Added `src/retrieval.py`, `src/answering.py`, `src/ollama_client.py`.
- Added `src/api.py` and `src/streamlit_app.py`.
- Added `src/evaluate_retrieval.py`, `tests/test_retrieval.py`, `tests/test_api.py`, and `tests/eval_questions.json`.
- Updated `requirements.txt` and `README.md`; added `.env.example`.
- Existing ingestion, processing, and index scripts and their data were preserved.

## Local run commands (PowerShell)

From the repository root, after Python 3.12 and the project dependencies are available:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.evaluate_retrieval --k 5
```

In one PowerShell window run Ollama (`ollama pull llama3.2:3b`, then `ollama serve`). In a second window:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In a third:

```powershell
streamlit run .\src\streamlit_app.py
```

The API can be checked with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/query `
  -ContentType "application/json" `
  -Body '{"question":"What does torch.Tensor.add do?"}'
```

## Verification status and known limitation

The existing workspace has the project source, 1,000-chunk corpus/index metadata, and an ignored `.venv` directory. In this execution environment, `.venv/pyvenv.cfg` points to `C:\Users\yadav\AppData\Local\Programs\Python\Python312\python.exe`, but that interpreter cannot be accessed here; invoking `.venv\Scripts\python.exe` reports that its configured base interpreter cannot be found. The Windows `python` and `py` commands do not resolve to a usable interpreter, and `ollama` is not available on PATH. I therefore could not run the newly added pytest suite, start FastAPI/Streamlit, download a model, or verify a live generated answer here. The report does not claim those runtime checks passed. Recreate the local virtual environment after Python 3.12 is available, then run the commands above; install/start Ollama separately and pull its model before testing answers.

No GitHub push or publication was performed.
