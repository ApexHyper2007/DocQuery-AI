"""Embed processed documentation chunks and store them in a local Chroma index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "index" / "chroma"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "docquery_docs"
UPSERT_BATCH_SIZE = 128

METADATA_FIELDS = (
    "source",
    "title_slug",
    "page_title",
    "configured_url",
    "source_url",
    "source_file",
    "heading_path",
    "section_title",
    "chunk_index",
    "chunk_count",
    "chunk_chars",
    "content_sha256",
)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load JSONL chunks and validate required fields and unique IDs."""
    chunks = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(chunk, dict):
            raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Chunk on line {line_number} is missing chunk_id")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Chunk {chunk_id} is missing non-empty content")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id in {path}: {chunk_id}")
        seen_ids.add(chunk_id)
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No chunks found in {path}; run src/process_docs.py first")
    return chunks


def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep the retrieval and citation fields in Chroma's scalar metadata format."""
    metadata: dict[str, str | int | float | bool] = {}
    for field in METADATA_FIELDS:
        value = chunk.get(field)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value
        elif isinstance(value, (list, dict)):
            metadata[field] = json.dumps(value, ensure_ascii=False)
        else:
            metadata[field] = str(value)
    return metadata


def build_index(
    input_path: Path = DEFAULT_INPUT,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    device: str = "cpu",
    batch_size: int = 32,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Encode the JSONL chunks and upsert them into a persistent cosine collection."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    chunks = load_chunks(input_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if rebuild:
        existing_collections = client.list_collections()
        collection_exists = any(
            candidate == collection_name
            or getattr(candidate, "name", None) == collection_name
            for candidate in existing_collections
        )
        if collection_exists:
            client.delete_collection(collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model and existing_model != model_name:
        raise ValueError(
            f"Collection {collection_name!r} uses {existing_model!r}, not {model_name!r}. "
            "Use --rebuild to recreate this project's collection with the selected model."
        )
    existing_metric = (collection.metadata or {}).get("hnsw:space")
    if existing_metric and existing_metric != "cosine":
        raise ValueError(
            f"Collection {collection_name!r} uses {existing_metric!r} distance, not 'cosine'. "
            "Use --rebuild to recreate this project's collection with cosine distance."
        )

    model_cache = PROJECT_ROOT / ".cache" / "sentence-transformers"
    model_cache.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device, cache_folder=str(model_cache))

    token_counts = [
        len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in (chunk["content"] for chunk in chunks)
    ]
    too_long = [index for index, count in enumerate(token_counts) if count > model.max_seq_length]
    if too_long:
        examples = ", ".join(
            f"{chunks[index]['chunk_id']} ({token_counts[index]} tokens)" for index in too_long[:5]
        )
        raise ValueError(
            f"{len(too_long)} chunk(s) exceed this model's {model.max_seq_length}-token limit and would be truncated: "
            f"{examples}. Re-run src/process_docs.py with a smaller --chunk-size, then index again."
        )

    dimension: int | None = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["content"] for chunk in batch]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if dimension is None:
            dimension = int(vectors.shape[1])
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=texts,
            metadatas=[_chroma_metadata(chunk) for chunk in batch],
            embeddings=vectors.tolist(),
        )
        batch_end = min(start + len(batch), len(chunks))
        print(f"Indexed chunks {start + 1}-{batch_end} of {len(chunks)}")

    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection_name,
        "persist_directory": str(persist_dir.relative_to(PROJECT_ROOT))
        if persist_dir.is_relative_to(PROJECT_ROOT)
        else str(persist_dir),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "distance_metric": "cosine",
        "device": device,
        "batch_size": batch_size,
        "chunk_file": str(input_path.relative_to(PROJECT_ROOT))
        if input_path.is_relative_to(PROJECT_ROOT)
        else str(input_path),
        "chunk_file_sha256": input_hash,
        "chunks_in_file": len(chunks),
        "chunks_in_collection": collection.count(),
    }
    info_path = persist_dir.parent / "index_info.json"
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Stored {len(chunks)} chunk(s) in collection {collection_name!r} "
        f"({dimension}-dimensional embeddings; {persist_dir})"
    )
    print(f"Index metadata: {info_path}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="processed chunks JSONL file")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete and recreate only this named DocQuery collection before indexing",
    )
    args = parser.parse_args()
    build_index(
        input_path=args.input,
        persist_dir=args.persist_dir,
        model_name=args.model,
        collection_name=args.collection,
        device=args.device,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
