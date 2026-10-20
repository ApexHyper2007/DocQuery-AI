"""Unit tests for documentation redirect resolution and response validation."""

import unittest
from unittest.mock import call, patch

from src import ingest_docs


def documentation_page(title: str = "Example API") -> bytes:
    text = "This is actual documentation content with enough useful details. " * 5
    return f"<html><head><title>{title}</title></head><body><main>{text}</main></body></html>".encode()


class RedirectDetectionTests(unittest.TestCase):
    base_url = "https://docs.pytorch.org/docs/2.14/guide/page.html"

    def test_javascript_location_replace_resolves_relative_destination(self) -> None:
        body = b"<title>Redirecting...</title><script>location.replace('../tensors.html')</script>"
        self.assertEqual(
            ingest_docs._redirect_target(body, self.base_url),
            "https://docs.pytorch.org/docs/2.14/tensors.html",
        )

    def test_javascript_redirect_with_hash_and_query_suffix(self) -> None:
        body = (
            b'<title>Redirecting...</title><script>'
            b'location.replace("../../2.14/generated/torch.Tensor.add.html" + location.hash + location.search)'
            b"</script>"
        )
        self.assertEqual(
            ingest_docs._redirect_target(body, "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.add.html"),
            "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.add.html",
        )

    def test_meta_refresh_resolves_relative_destination(self) -> None:
        body = b'<title>Moving</title><meta http-equiv="refresh" content="0; url=../tensors.html">'
        self.assertEqual(
            ingest_docs._redirect_target(body, self.base_url),
            "https://docs.pytorch.org/docs/2.14/tensors.html",
        )

    def test_canonical_and_anchor_links_are_redirect_fallbacks(self) -> None:
        canonical = b'<title>Redirecting...</title><link rel="canonical" href="../generated/api.html">'
        anchor = b'<title>Redirecting...</title><p>Redirecting</p><a href="../guide.html">Continue</a>'
        self.assertEqual(
            ingest_docs._redirect_target(canonical, self.base_url),
            "https://docs.pytorch.org/docs/2.14/generated/api.html",
        )
        self.assertEqual(
            ingest_docs._redirect_target(anchor, self.base_url),
            "https://docs.pytorch.org/docs/2.14/guide.html",
        )

    def test_fetch_follows_html_redirect_and_uses_response_final_url_as_base(self) -> None:
        stub = b"<title>Redirecting...</title><script>location.replace('v2/page.html')</script>"
        final_url = "https://docs.pytorch.org/docs/stable/landing.html"
        resolved_url = "https://docs.pytorch.org/docs/stable/v2/page.html"
        with patch.object(
            ingest_docs,
            "_read_response",
            side_effect=[(stub, "text/html", 200, final_url), (documentation_page(), "text/html", 200, resolved_url)],
        ) as read_response:
            result = ingest_docs.fetch_page("https://docs.pytorch.org/old/page.html")
        self.assertEqual(result[0], documentation_page())
        self.assertEqual(result[3], resolved_url)
        self.assertEqual(
            read_response.call_args_list,
            [call("https://docs.pytorch.org/old/page.html"), call(resolved_url)],
        )

    def test_unresolved_redirect_stub_is_rejected(self) -> None:
        stub = b"<html><head><title>Redirecting...</title></head><body>Redirecting</body></html>"
        with patch.object(ingest_docs, "_read_response", return_value=(stub, "text/html", 200, self.base_url)):
            with self.assertRaisesRegex(ingest_docs.IngestionError, "Unresolved HTML redirect stub"):
                ingest_docs.fetch_page(self.base_url)

    def test_redirect_loop_is_rejected(self) -> None:
        stub = b"<title>Redirecting...</title><script>location.replace('/docs/2.14/guide/page.html')</script>"
        with patch.object(ingest_docs, "_read_response", return_value=(stub, "text/html", 200, self.base_url)):
            with self.assertRaisesRegex(ingest_docs.IngestionError, "redirect loop"):
                ingest_docs.fetch_page(self.base_url)


if __name__ == "__main__":
    unittest.main()
