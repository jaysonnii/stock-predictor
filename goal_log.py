"""
Permanent log of the daily $1M check ins: date, net worth, what you said you
would do today, and the core belief shown that day.

Saved one JSON line per check in at goal_data/checkins.jsonl on this computer.
goal_data/ is in .gitignore so your net worth never goes to GitHub.
"""
import csv
import io
import json
import os
import re
import threading

LOG_DIR = "goal_data"
LOG_FILE = "checkins.jsonl"
MAX_ACTION = 300
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_lock = threading.Lock()


def _path():
    return os.path.join(LOG_DIR, LOG_FILE)


def clean_entry(raw) -> dict:
    """Validate one check in. Raises ValueError with a readable message."""
    if not isinstance(raw, dict):
        raise ValueError("Each check in must be an object.")
    date = str(raw.get("date", ""))
    if not _DATE.match(date):
        raise ValueError("date must look like 2026-11-04.")
    try:
        net_worth = float(raw.get("netWorth"))
    except (TypeError, ValueError):
        raise ValueError("netWorth must be a number.")
    if not (0 <= net_worth < 1e12):
        raise ValueError("netWorth is out of range.")
    action = " ".join(str(raw.get("action", "")).split())
    if not action:
        raise ValueError("action is empty.")
    belief = " ".join(str(raw.get("belief") or "").split())[:80] or None
    return {"date": date, "netWorth": round(net_worth, 2), "action": action[:MAX_ACTION], "belief": belief}


def _key(e):
    return (e["date"], e["action"])


def read_all() -> list:
    try:
        with open(_path(), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(clean_entry(json.loads(line)))
        except (ValueError, json.JSONDecodeError):
            continue  # skip a damaged line rather than lose the whole log
    out.sort(key=lambda e: e["date"])
    return out


def append(entries) -> dict:
    """Add check ins that are not already saved (same date and same words). Returns counts."""
    new = [clean_entry(e) for e in entries]
    with _lock:
        existing = {_key(e) for e in read_all()}
        added = []
        for e in new:
            if _key(e) not in existing:
                existing.add(_key(e))
                added.append(e)
        if added:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(_path(), "a", encoding="utf-8") as f:
                for e in added:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return {"added": len(added), "total": len(existing)}


def to_csv(entries) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "net_worth", "what_i_will_do_today", "belief"])
    for e in entries:
        w.writerow([e["date"], e["netWorth"], e["action"], e["belief"] or ""])
    return buf.getvalue()
