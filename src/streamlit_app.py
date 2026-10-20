"""Streamlit chat interface for the DocQuery FastAPI backend."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


st.set_page_config(page_title="DocQuery AI", page_icon="📚", layout="wide")
st.title("DocQuery AI")
st.caption("Ask PyTorch, torchvision, and CUDA questions grounded in official documentation.")
backend_url = st.sidebar.text_input(
    "FastAPI URL", os.getenv("DOCQUERY_API_URL", "http://localhost:8000")
).rstrip("/")
st.sidebar.caption("The answer service requires Ollama to be running locally.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Retrieved sources ({len(sources)})", expanded=True):
        for source in sources:
            title = source.get("section_title") or source.get("page_title") or source.get("source", "Documentation")
            citation = source.get("citation", "")
            url = source.get("source_url") or source.get("configured_url") or ""
            st.markdown(f"**[{citation}] {title}**  \n[{url}]({url})")
            with st.container(border=True):
                st.caption(source.get("content", ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask a technical documentation question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and drafting a cited answer…"):
            payload = json.dumps({"question": question}).encode("utf-8")
            request = Request(
                f"{backend_url}/api/query", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urlopen(request, timeout=180) as response:
                    result = json.loads(response.read().decode("utf-8"))
                answer = result["answer"]
                st.markdown(answer)
                if result.get("insufficient_context"):
                    st.warning("The indexed excerpts may not contain enough evidence for an answer.")
                if result.get("sources"):
                    show_sources(result["sources"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": result.get("sources", [])}
                )
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                st.error(f"Backend error ({exc.code}): {detail}")
            except (URLError, TimeoutError, OSError) as exc:
                st.error(f"Cannot reach the FastAPI backend at {backend_url}. Start the API first. ({exc})")
            except (KeyError, json.JSONDecodeError) as exc:
                st.error(f"The backend returned an invalid response: {exc}")
