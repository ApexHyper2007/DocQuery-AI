"""Download a small, explicit set of official documentation pages as raw HTML."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAX_PAGE_BYTES = 15 * 1024 * 1024
MAX_HTML_REDIRECTS = 5

# A small starter set: API, usage guide, and foundational GPU concepts.
SOURCES = (
    ("pytorch", "tensor-add", "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.add.html"),
    ("pytorch", "tensor-guide", "https://docs.pytorch.org/docs/stable/tensors.html"),
    ("torchvision", "transforms", "https://docs.pytorch.org/vision/stable/transforms.html"),
    ("torchvision", "datasets", "https://docs.pytorch.org/vision/stable/datasets.html"),
    ("cuda", "programming-model", "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html"),
    ("cuda", "cuda-platform", "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html"),
)

ALLOWED_HOSTS = {"docs.pytorch.org", "docs.nvidia.com"}
USER_AGENT = "DocQuery-AI/0.1 (documentation research project)"
_JS_REDIRECT = re.compile(
    r"(?:window\s*\.\s*)?location\s*\.\s*replace\s*\(\s*(['\"])(.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
_META_REFRESH = re.compile(r"(?:^|;)\s*url\s*=\s*(['\"]?)(.*?)\1\s*$", re.IGNORECASE)


class IngestionError(RuntimeError):
    """A download or validation failure with response details for the manifest."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        final_url: str | None = None,
        content_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.final_url = final_url
        self.content_type = content_type


class _RedirectMarkup(HTMLParser):
    """Extract redirect clues and enough visible text to reject empty stubs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.script_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.meta_refresh: list[str] = []
        self._in_title = False
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "title":
            self._in_title = True
        elif tag.lower() == "script":
            self._in_script = True
        elif tag.lower() == "style":
            self._in_style = True
        elif tag.lower() == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            self.meta_refresh.append(attributes.get("content", ""))
        elif tag.lower() == "link":
            rel = attributes.get("rel", "").lower().split()
            href = attributes.get("href", "").strip()
            if "canonical" in rel and href:
                self.links.append(("canonical", href))
        elif tag.lower() == "a":
            href = attributes.get("href", "").strip()
            if href:
                self.links.append(("anchor", href))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() == "script":
            self._in_script = False
        elif tag.lower() == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_script:
            self.script_parts.append(data)
        elif not self._in_style and not self._in_title:
            self.text_parts.append(data)


def _parse_markup(body: bytes) -> _RedirectMarkup:
    parser = _RedirectMarkup()
    parser.feed(body.decode("utf-8", errors="replace"))
    return parser


def _redirect_target(body: bytes, final_url: str) -> str | None:
    """Return a resolved HTML redirect destination, if the page is a redirect stub."""
    parser = _parse_markup(body)
    title = " ".join(" ".join(parser.title_parts).split()).lower()
    visible_text = " ".join(" ".join(parser.text_parts).split())

    for content in parser.meta_refresh:
        match = _META_REFRESH.search(content)
        if match and match.group(2).strip():
            return urljoin(final_url, match.group(2).strip())

    script_text = " ".join(parser.script_parts)
    match = _JS_REDIRECT.search(script_text)
    if match:
        return urljoin(final_url, match.group(2).strip())

    redirect_title = title.startswith("redirecting") or title.startswith("redirect")
    if redirect_title or (len(visible_text) < 300 and "redirect" in visible_text.lower()):
        for kind in ("canonical", "anchor"):
            for link_kind, href in parser.links:
                if link_kind == kind:
                    return urljoin(final_url, href)
    return None


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise IngestionError(f"Refusing destination outside the HTTPS documentation allowlist: {url}")


def _read_response(url: str) -> tuple[bytes, str, int, str]:
    """Read one page; urllib follows standard HTTP redirects automatically."""
    _validate_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        response = urlopen(request, timeout=30)
    except HTTPError as exc:
        raise IngestionError(
            f"HTTP request failed with status {exc.code} for {url}",
            http_status=exc.code,
            final_url=exc.geturl(),
            content_type=exc.headers.get("Content-Type", ""),
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise IngestionError(f"Request failed for {url}: {exc}") from exc

    with response:
        final_url = response.geturl()
        try:
            _validate_url(final_url)
        except IngestionError as exc:
            raise IngestionError(str(exc), final_url=final_url) from exc
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            raise IngestionError(
                f"Expected HTML at {final_url}; received {content_type!r}",
                http_status=response.status,
                final_url=final_url,
                content_type=content_type,
            )
        body = response.read(MAX_PAGE_BYTES + 1)
        if len(body) > MAX_PAGE_BYTES:
            raise IngestionError(
                f"Page exceeds the {MAX_PAGE_BYTES}-byte safety limit: {final_url}",
                http_status=response.status,
                final_url=final_url,
                content_type=content_type,
            )
        return body, content_type, response.status, final_url


def fetch_page(url: str) -> tuple[bytes, str, int, str]:
    """Fetch and validate a documentation page, resolving HTML redirect stubs."""
    current_url = url
    visited: set[str] = set()

    for redirect_count in range(MAX_HTML_REDIRECTS + 1):
        if current_url in visited:
            raise IngestionError(f"HTML redirect loop detected at {current_url}", final_url=current_url)
        visited.add(current_url)
        body, content_type, status, final_url = _read_response(current_url)

        # urllib may have followed a normal HTTP redirect. Track that URL too so
        # an HTML stub cannot send the request back into the same chain.
        if final_url in visited and final_url != current_url:
            raise IngestionError(f"HTTP/HTML redirect loop detected at {final_url}", final_url=final_url)
        visited.add(final_url)

        target = _redirect_target(body, final_url)
        if target is None:
            parser = _parse_markup(body)
            title = " ".join(" ".join(parser.title_parts).split()).lower()
            text = " ".join(parser.text_parts).strip()
            if title.startswith("redirect") or (len(text) < 100 and "redirect" in text.lower()):
                raise IngestionError(
                    f"Unresolved HTML redirect stub at {final_url}",
                    http_status=status,
                    final_url=final_url,
                    content_type=content_type,
                )
            if len(text) < 100:
                raise IngestionError(
                    f"Response does not contain enough documentation text: {final_url}",
                    http_status=status,
                    final_url=final_url,
                    content_type=content_type,
                )
            return body, content_type, status, final_url

        if redirect_count >= MAX_HTML_REDIRECTS:
            raise IngestionError(
                f"Exceeded the limit of {MAX_HTML_REDIRECTS} HTML redirects from {url}",
                http_status=status,
                final_url=final_url,
                content_type=content_type,
            )
        if target in visited:
            raise IngestionError(f"HTML redirect loop detected: {final_url} -> {target}", final_url=final_url)
        _validate_url(target)
        current_url = target

    raise IngestionError(f"Could not resolve documentation page: {url}")


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Manifest is not valid JSON: {path}") from exc
    if not isinstance(content, list):
        raise RuntimeError(f"Manifest must contain a JSON list: {path}")
    entries: dict[str, dict[str, Any]] = {}
    for item in content:
        if isinstance(item, dict):
            key = item.get("configured_url") or item.get("url")
            if key:
                entries[key] = item
    return entries


def ingest(refresh: bool = False) -> int:
    """Save starter pages and a JSON manifest; return the number fetched successfully."""
    manifest_path = RAW_DIR / "manifest.json"
    manifest_by_configured_url = _load_manifest(manifest_path)

    fetched_count = 0
    for source, slug, configured_url in SOURCES:
        output_path = RAW_DIR / source / f"{slug}.html"
        existing = manifest_by_configured_url.get(configured_url, {})
        if output_path.exists() and not refresh and existing.get("status", "success") == "success":
            print(f"SKIP (already exists): {output_path.relative_to(PROJECT_ROOT)}")
            continue

        try:
            body, content_type, status, final_url = fetch_page(configured_url)
        except IngestionError as exc:
            previous_success = None
            if existing.get("status", "success") == "success" and existing:
                previous_success = {
                    key: existing[key]
                    for key in ("final_url", "url", "http_status", "content_type", "bytes", "sha256", "fetched_at_utc")
                    if key in existing
                }
            failed_entry = {
                "source": source,
                "title_slug": slug,
                "configured_url": configured_url,
                "url": exc.final_url or configured_url,
                "final_url": exc.final_url,
                "file": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "http_status": exc.http_status,
                "content_type": exc.content_type,
                "error": str(exc),
            }
            if previous_success:
                failed_entry["previous_success"] = previous_success
            manifest_by_configured_url[configured_url] = failed_entry
            print(f"ERROR {configured_url}: {exc}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(body)
        manifest_by_configured_url[configured_url] = {
            "source": source,
            "title_slug": slug,
            "configured_url": configured_url,
            "url": final_url,
            "final_url": final_url,
            "file": output_path.relative_to(PROJECT_ROOT).as_posix(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "http_status": status,
            "content_type": content_type,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        fetched_count += 1
        print(f"SAVED {output_path.relative_to(PROJECT_ROOT)} ({len(body):,} bytes) from {final_url}")
        time.sleep(1)

    updated_manifest = sorted(
        manifest_by_configured_url.values(),
        key=lambda item: item.get("configured_url", item.get("url", "")),
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(updated_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    return fetched_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download pages again even when their local HTML file already exists",
    )
    args = parser.parse_args()
    count = ingest(refresh=args.refresh)
    print(f"Fetched {count} page(s).")


if __name__ == "__main__":
    main()
