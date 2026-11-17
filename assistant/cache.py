"""Route 0: exact repeats are answered from a store, with no model call.

A request is normalised (case, spacing, punctuation at the ends) and hashed.
Success criterion H2: an exact repeat costs 0 tokens. A similarity cache
(close but not identical requests) is left for after the pilot, because its
threshold (config.CLOSE_MATCH_COSINE) must be set from pilot data first.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional


def normalise(request: str) -> str:
    text = re.sub(r"\s+", " ", request.strip().lower())
    return text.strip(" ?!.")


def key(request: str) -> str:
    return hashlib.sha256(normalise(request).encode("utf-8")).hexdigest()[:16]


class ExactCache:
    """A JSON file of {key: {"request", "answer", "kind", "commit", "time"}}.

    Entries carry the commit of the knowledge base they were made from, so
    a cached answer can be ignored once the notebooks or code change.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.store = {}
        if self.path and self.path.exists():
            self.store = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, request: str, commit: str = "") -> Optional[dict]:
        entry = self.store.get(key(request))
        if entry is None:
            return None
        if commit and entry.get("commit") and entry["commit"] != commit:
            return None  # stale: made from an older knowledge base
        return entry

    def put(self, request: str, answer: str, kind: str, commit: str = "") -> None:
        self.store[key(request)] = {
            "request": request, "answer": answer, "kind": kind, "commit": commit, "time": time.time(),
        }
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.store, indent=1), encoding="utf-8")
