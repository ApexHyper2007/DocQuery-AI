"""Clean downloaded documentation HTML and split it into metadata-rich chunks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MANIFEST = RAW_DIR / "manifest.json"
DEFAULT_OUTPUT = PROCESSED_DIR / "chunks.jsonl"

CONTENT_ROOT_SELECTORS = (
    "main#main-content",
    "article#pytorch-article",
    "article[itemprop='articleBody']",
    "article.bd-article",
    "main",
)
REMOVE_TAGS = ("script", "style", "nav", "footer", "noscript", "svg", "form", "button")
CODE_FENCE_RE = re.compile(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


def _render_table(table: Tag, final_url: str) -> str:
    rows: list[list[str]] = []
    header_row: int | None = None
    for row in table.find_all("tr"):
        cells = row.find_all(("th", "td"), recursive=False)
        if not cells:
            continue
        if header_row is None and any(cell.name == "th" for cell in cells):
            header_row = len(rows)
        rows.append([render_html(cell, final_url).replace("\n", " ").strip().replace("|", r"\|") for cell in cells])
    if not rows:
        return ""
    if header_row is None:
        header_row = 0
    else:
        rows = [rows[header_row], *rows[:header_row], *rows[header_row + 1 :]]
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    rendered = ["| " + " | ".join(rows[0]) + " |"]
    rendered.append("| " + " | ".join("---" for _ in range(width)) + " |")
    rendered.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(rendered) + "\n\n"


def render_html(node: Any, final_url: str) -> str:
    """Convert useful HTML structure to Markdown while retaining code and links."""
    if isinstance(node, NavigableString):
        text = str(node)
        return "" if not text.strip() else text
    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name in REMOVE_TAGS or "hidden" in node.attrs or node.get("aria-hidden") == "true":
        return ""
    classes = node.get("class", [])
    if any(class_name in {"headerlink", "copybtn", "viewcode-link"} for class_name in classes):
        return ""

    if name == "pre":
        code_node = node.find("code") or node
        code = code_node.get_text().strip("\n")
        code_classes = " ".join(code_node.get("class", []))
        language_match = re.search(r"(?:language|highlight)-([\w+#.-]+)", code_classes)
        language = language_match.group(1) if language_match else ""
        longest_ticks = max((len(match.group(0)) for match in re.finditer(r"`+", code)), default=0)
        fence = "`" * max(3, longest_ticks + 1)
        return f"\n\n{fence}{language}\n{code}\n{fence}\n\n"
    if name == "table":
        return _render_table(node, final_url)
    if name in {"ul", "ol"}:
        items = node.find_all("li", recursive=False)
        rendered_items = []
        for index, item in enumerate(items, start=1):
            marker = f"{index}." if name == "ol" else "-"
            rendered_items.append(f"{marker} {render_html(item, final_url).strip()}")
        return "\n".join(rendered_items) + "\n\n"

    content = "".join(render_html(child, final_url) for child in node.children)
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        return f"\n\n{'#' * level} {content.strip()}\n\n"
    if name == "a":
        href = node.get("href")
        label = content.strip()
        if href and label:
            return f"[{label}]({urljoin(final_url, href)})"
        return label
    if name == "code":
        label = content.strip()
        if not label:
            return ""
        ticks = "`" * (max((len(match.group(0)) for match in re.finditer(r"`+", label)), default=0) + 1)
        return f"{ticks}{label}{ticks}"
    if name == "br":
        return "\n"
    if name in {"p", "div", "section", "article", "main", "blockquote", "figure", "dl", "dt", "dd"}:
        return f"\n\n{content.strip()}\n\n" if content.strip() else ""
    return content


def clean_html(html: bytes, final_url: str) -> tuple[str, str]:
    """Remove site furniture and return (document title, structured Markdown)."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for selector in ("a.headerlink", ".copybtn", ".viewcode-link"):
        for element in soup.select(selector):
            element.decompose()

    content_root = None
    for selector in CONTENT_ROOT_SELECTORS:
        content_root = soup.select_one(selector)
        if content_root is not None:
            break
    if content_root is None:
        content_root = soup.body or soup

    for tag in content_root.find_all(REMOVE_TAGS):
        tag.decompose()
    for aside in content_root.find_all("aside"):
        aside_identity = " ".join(
            [aside.get("id", ""), " ".join(aside.get("class", [])), aside.get("role", "")]
        ).lower()
        if re.search(r"sidebar|side-nav|menu|version-warning|complementary", aside_identity):
            aside.decompose()
    markdown = render_html(content_root, final_url)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n[ \t]+", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if not title:
        first_heading = re.search(r"(?m)^#\s+(.+)$", markdown)
        title = first_heading.group(1).strip() if first_heading else "Untitled documentation page"
    return title, markdown


def _split_sections(markdown: str, page_title: str) -> list[tuple[list[tuple[int, str]], str]]:
    """Split at Markdown headings, without treating headings inside code as sections."""
    sections: list[tuple[list[tuple[int, str]], str]] = []
    heading_path: list[tuple[int, str]] = []
    body_lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body or heading_path:
            path = heading_path.copy() or [(1, page_title)]
            sections.append((path, body))
        body_lines.clear()

    for line in markdown.splitlines():
        fence_match = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
        if not in_fence:
            heading_match = HEADING_RE.match(line)
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                del heading_path[level - 1 :]
                heading_path.append((level, heading_match.group(2).strip()))
                continue
        if fence_match:
            if not in_fence:
                in_fence = True
            elif line.strip().startswith(fence_match.group(1)[0] * len(fence_match.group(1))):
                in_fence = False
        body_lines.append(line)
    flush()
    if not sections:
        sections.append(([(1, page_title)], markdown.strip()))
    return sections


def _split_preserving_code(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split prose recursively and keep fenced examples intact where possible."""
    segments: list[tuple[bool, str]] = []
    cursor = 0
    for match in CODE_FENCE_RE.finditer(text):
        if text[cursor : match.start()].strip():
            segments.append((False, text[cursor : match.start()].strip()))
        segments.append((True, match.group(0)))
        cursor = match.end()
    if text[cursor:].strip():
        segments.append((False, text[cursor:].strip()))

    atoms: list[str] = []
    for is_code, segment in segments:
        if not is_code:
            atoms.extend(splitter.split_text(segment))
            continue
        if len(segment) <= chunk_size:
            atoms.append(segment)
            continue

        opening_line, code = segment.split("\n", maxsplit=1)
        fence = CODE_FENCE_RE.match(segment).group(1)  # type: ignore[union-attr]
        code = re.sub(rf"\n{re.escape(fence)}[ \t]*$", "", code)
        inner_size = max(100, chunk_size - len(opening_line) - len(fence) - 2)
        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=inner_size,
            chunk_overlap=min(chunk_overlap, max(0, inner_size // 4)),
            separators=["\n", " ", "\t"],
        )
        code_parts = code_splitter.split_text(code)
        atoms.extend(f"{opening_line}\n{part}\n{fence}" for part in code_parts)

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = f"{current}\n\n{atom}" if current else atom
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            current = atom
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _safe_manifest_file(relative_file: str) -> Path:
    raw_root = RAW_DIR.resolve()
    path = (PROJECT_ROOT / relative_file).resolve()
    if path != raw_root and raw_root not in path.parents:
        raise ValueError(f"Manifest file path is outside data/raw: {relative_file}")
    return path


def process_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    chunk_size: int = 450,
    chunk_overlap: int = 60,
) -> int:
    """Clean every successful manifest page and write chunks as JSON Lines."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest must contain a JSON list: {manifest_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    pages: list[dict[str, Any]] = []
    for entry in manifest:
        if not isinstance(entry, dict) or entry.get("status", "success") != "success":
            continue
        relative_file = entry.get("file")
        final_url = entry.get("final_url") or entry.get("url")
        if not relative_file or not final_url:
            raise ValueError(f"Manifest entry is missing file or final URL: {entry}")
        html_path = _safe_manifest_file(relative_file)
        if not html_path.is_file():
            raise FileNotFoundError(f"Downloaded documentation file is missing: {html_path}")
        title, markdown = clean_html(html_path.read_bytes(), final_url)
        if len(markdown) < 100:
            raise ValueError(f"Cleaned page has too little documentation content: {html_path}")
        pages.append({"entry": entry, "title": title, "markdown": markdown, "path": html_path})

    if not pages:
        raise ValueError(f"No successful documentation pages found in {manifest_path}")

    records: list[dict[str, Any]] = []
    for page in pages:
        entry = page["entry"]
        source = entry.get("source", "unknown")
        slug = entry.get("title_slug", page["path"].stem)
        final_url = entry.get("final_url") or entry.get("url")
        page_sections = _split_sections(page["markdown"], page["title"])
        page_chunks: list[tuple[list[tuple[int, str]], str]] = []
        for headings, body in page_sections:
            heading_prefix = "\n".join(f"{'#' * level} {text}" for level, text in headings)
            section_budget = max(100, chunk_size - len(heading_prefix) - 2)
            section_splitter = RecursiveCharacterTextSplitter(
                chunk_size=section_budget,
                chunk_overlap=min(chunk_overlap, max(0, section_budget // 4)),
                separators=["\n\n", "\n", " "],
            )
            section_chunks = _split_preserving_code(body, section_splitter, section_budget, chunk_overlap)
            if not section_chunks:
                section_chunks = [""]
            for section_chunk in section_chunks:
                content = f"{heading_prefix}\n\n{section_chunk}".strip()
                page_chunks.append((headings, content))

        for chunk_index, (headings, content) in enumerate(page_chunks):
            heading_names = [text for _, text in headings]
            chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            stable_id = hashlib.sha256(
                f"{final_url}\0{chunk_index}\0{chunk_hash}".encode("utf-8")
            ).hexdigest()[:20]
            records.append(
                {
                    "chunk_id": f"{source}-{slug}-{stable_id}",
                    "source": source,
                    "title_slug": slug,
                    "page_title": page["title"],
                    "configured_url": entry.get("configured_url", final_url),
                    "source_url": final_url,
                    "source_file": entry["file"],
                    "heading_path": heading_names,
                    "section_title": heading_names[-1] if heading_names else page["title"],
                    "chunk_index": chunk_index,
                    "chunk_count": len(page_chunks),
                    "chunk_chars": len(content),
                    "content_sha256": chunk_hash,
                    "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "content": content,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Processed {len(pages)} page(s) into {len(records)} chunk(s): {output_path}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=450, help="maximum target chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=60, help="target overlap between chunks in characters")
    args = parser.parse_args()
    process_manifest(args.manifest, args.output, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
