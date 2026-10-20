"""Small Ollama HTTP client with no additional Python dependency."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


class OllamaError(RuntimeError):
    """Ollama is unavailable or returned an invalid response."""


def generate_grounded_answer(
    question: str,
    sources: list[dict[str, Any]],
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Ask Ollama to answer only from supplied snippets and cite [S#] labels."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not sources:
        raise ValueError("At least one retrieved source is required")
    model = model or os.getenv("DOCQUERY_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = (base_url or os.getenv("DOCQUERY_OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
    context_parts = []
    for number, source in enumerate(sources, start=1):
        title = source.get("section_title") or source.get("page_title") or source.get("source", "Documentation")
        url = source.get("source_url") or source.get("configured_url") or ""
        context_parts.append(f"[S{number}] source={title}; url={url}\n{source.get('content', '')}")
    system = (
        "You are DocQuery AI, a technical documentation assistant. Answer using only the supplied "
        "documentation excerpts. Cite each factual claim with one or more source labels like [S1]. "
        "If the excerpts do not support an answer, say that the available documentation is insufficient. "
        "Do not invent API behavior, parameters, versions, or citations. Keep code examples faithful to the excerpts."
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Documentation excerpts:\n\n" + "\n\n".join(context_parts) + "\n\nQuestion: " + question},
        ],
        "options": {"temperature": 0.1},
    }
    request = Request(
        f"{base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}. Start Ollama and pull model {model!r}. ({exc})"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaError("Ollama returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise OllamaError("Ollama returned an invalid response object")
    message_object = result.get("message")
    if not isinstance(message_object, dict):
        raise OllamaError("Ollama response did not contain a message object")
    message = message_object.get("content")
    if not isinstance(message, str) or not message.strip():
        raise OllamaError("Ollama returned an empty answer")
    return message.strip()
