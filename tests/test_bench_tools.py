from agent_harness.bench.tools import web_fetch


def test_non_ascii_urls_are_encoded_not_crashing(monkeypatch):
    seen = {}

    class R:
        headers = type("H", (), {"get_content_charset": lambda s: "utf-8", "get": lambda s, k, d="": "text/plain"})()
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n): return b"ok"

    class Op:
        def open(self, req, timeout=0):
            seen["url"] = req.full_url
            return R()
    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Op())
    assert web_fetch("https://example.com/a b/café?q=…") == "ok"
    assert seen["url"].isascii() and "%E2%80%A6" in seen["url"]
