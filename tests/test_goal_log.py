import json

import pytest
from fastapi.testclient import TestClient

import goal_log
import main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(goal_log, "LOG_DIR", str(tmp_path / "goal_data"))
    return tmp_path


def entry(date="2026-11-04", nw=312400, action="Move $800 to the brokerage", belief="Show up every day"):
    return {"date": date, "netWorth": nw, "action": action, "belief": belief}


def test_empty_log():
    assert client.get("/goal/checkins").json() == {"count": 0, "checkins": []}


def test_save_read_and_skip_duplicates():
    r = client.post("/goal/checkins", json=[entry(), entry("2026-11-03", 310000, "Call two sellers about the fourplex")])
    assert r.json() == {"added": 2, "total": 2}
    # same day and same words, extra spaces included, is not saved twice
    r = client.post("/goal/checkins", json=[entry(action="Move  $800 to the   brokerage")])
    assert r.json() == {"added": 0, "total": 2}
    # same day, different words, is a second answer and is kept
    assert client.post("/goal/checkins", json=[entry(action="Also list the GSA truck")]).json()["added"] == 1
    data = client.get("/goal/checkins").json()
    assert data["count"] == 3
    assert [c["date"] for c in data["checkins"]] == ["2026-11-03", "2026-11-04", "2026-11-04"]


def test_rejects_bad_entries():
    assert client.post("/goal/checkins", json=[entry(date="11/04/2026")]).status_code == 400
    assert client.post("/goal/checkins", json=[entry(nw="lots")]).status_code == 400
    assert client.post("/goal/checkins", json=[entry(action="   ")]).status_code == 400
    assert client.get("/goal/checkins").json()["count"] == 0  # nothing partial was written


def test_long_action_is_trimmed_and_damaged_lines_skipped(tmp_log):
    client.post("/goal/checkins", json=[entry(action="x" * 500)])
    path = tmp_log / "goal_data" / "checkins.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write("{not json\n")
    data = client.get("/goal/checkins").json()
    assert data["count"] == 1 and len(data["checkins"][0]["action"]) == goal_log.MAX_ACTION
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["netWorth"] == 312400


def test_csv_download():
    client.post("/goal/checkins", json=[entry(action='Said "no" to a new car, kept the cash')])
    r = client.get("/goal/checkins.csv")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    lines = r.text.strip().splitlines()
    assert lines[0] == "date,net_worth,what_i_will_do_today,belief"
    assert lines[1].startswith('2026-11-04,312400.0,"Said ""no"" to a new car, kept the cash"')
