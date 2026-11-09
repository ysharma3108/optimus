"""Log the real token use of every model call, one JSON line per call.

Reza asked for measured, not estimated, token counts. The counts written
here come from the `usage` block the Anthropic Messages API returns with
every response (input_tokens, output_tokens, cache_creation_input_tokens,
cache_read_input_tokens), or from Ollama's prompt_eval_count / eval_count.
Nothing here estimates tokens.

The pilot reads this log to replace the assumptions in the cost model.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class CallRecord:
    model: str
    route: str  # "question", "template", "new_script", "fix", "restart", ...
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0
    outcome: str = ""  # "ok", "no_citation", "run_failed", ...
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)


class TokenLog:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.records: List[CallRecord] = []
        if self.path and self.path.exists():
            self.records = list(read_log(self.path))

    def add(self, record: CallRecord) -> None:
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    def summary(self) -> Dict[str, dict]:
        return summarise(self.records)


def read_log(path: Path) -> Iterable[CallRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield CallRecord(**json.loads(line))


def summarise(records: Iterable[CallRecord]) -> Dict[str, dict]:
    """Per (route, model): number of calls, mean and total tokens, mean time."""
    groups: Dict[str, List[CallRecord]] = {}
    for r in records:
        groups.setdefault(r.route + " | " + r.model, []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        out[key] = {
            "calls": len(rs),
            "mean_input": statistics.fmean(r.input_tokens for r in rs),
            "mean_output": statistics.fmean(r.output_tokens for r in rs),
            "total_input": sum(r.input_tokens for r in rs),
            "total_output": sum(r.output_tokens for r in rs),
            "total_cache_read": sum(r.cache_read_tokens for r in rs),
            "total_cache_write": sum(r.cache_write_tokens for r in rs),
            "mean_seconds": statistics.fmean(r.seconds for r in rs),
        }
    return out


def cost_usd(record: CallRecord, prices: Dict[str, Dict[str, float]]) -> float:
    """Cost of one call from a price table in USD per million tokens:
    {"model": {"input": .., "output": .., "cache_read": .., "cache_write": ..}}.

    Prices are passed in, not stored here, so they always come from the
    dated, cited table in the cost model.
    """
    p = prices[record.model]
    return (
        record.input_tokens * p["input"]
        + record.output_tokens * p["output"]
        + record.cache_read_tokens * p.get("cache_read", p["input"])
        + record.cache_write_tokens * p.get("cache_write", p["input"])
    ) / 1e6
