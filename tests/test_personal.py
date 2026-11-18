import json
import re

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def _personal(r):
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"
    prefix = "window.PERSONAL = "
    assert r.text.startswith(prefix) and r.text.endswith(";")
    return json.loads(r.text[len(prefix):-1])


def test_blank_when_no_personal_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PERSONAL_PATH", str(tmp_path / "personal.json"))
    assert _personal(client.get("/personal.js")) == {}


def test_serves_personal_file(tmp_path, monkeypatch):
    path = tmp_path / "personal.json"
    path.write_text(json.dumps({"holdings": [{"ticker": "VOO", "shares": 1, "costBasis": 500}],
                                "beliefs": ["</script><b>x</b>"]}), encoding="utf-8")
    monkeypatch.setattr(main, "PERSONAL_PATH", str(path))
    data = _personal(client.get("/personal.js"))
    assert data["holdings"][0]["ticker"] == "VOO" and data["beliefs"] == ["</script><b>x</b>"]


def test_broken_file_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "personal.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(main, "PERSONAL_PATH", str(path))
    assert _personal(client.get("/personal.js")) == {}


def test_public_code_has_no_personal_data():
    """Holdings, beliefs and goal dates must come from personal.json, never be written into the page."""
    html = open("static/index.html", encoding="utf-8").read()
    assert re.search(r"SEED_HOLDINGS\s*=\s*\[", html) is None          # no hard-coded holdings list
    assert re.search(r"\{\s*ticker:\s*'[A-Z.]+',\s*shares:\s*[0-9]", html) is None
    assert re.search(r"CORE_BELIEFS\s*=\s*\[\s*['\"]", html) is None   # no hard-coded beliefs
    assert re.search(r"deadline:\s*'\d{4}-\d{2}-\d{2}'", html) is None   # no hard-coded goal date
    assert "getFullYear() + 7" not in html                                  # no made-up default deadline
