"""Pick one of the four routes for a request (hybrid design of 3 Nov 2026).

  repeat      route 0: the same request was answered before -> stored answer
  question    route 1: a how/what/why question -> cited answer from the docs
  template    route 2: a script close to a verified example -> fill values
  new_script  route 3: a script with no close match -> write, check, run, fix

The first version is plain rules, so it costs no tokens and is easy to
explain. Router accuracy is measured in the pilot (success criterion H1);
a model classifier replaces the rules only if they fall short.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from . import config
from .cache import ExactCache
from .retrieve import Retriever, tokenize

REPEAT, QUESTION, TEMPLATE, NEW_SCRIPT = "repeat", "question", "template", "new_script"

# Words that ask for a runnable script rather than an explanation.
_SCRIPT = re.compile(
    r"\b(write|generate|give me|create|make|build|produce|set up|simulate|run)\b.*"
    r"\b(script|code|simulation|model|example|program)\b"
    r"|\b(script|code)\s+(for|to|that)\b"
    r"|^\s*simulate\b",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(r"^\s*(what|why|which|when|where|who|is|are|does|do|can|explain|how does|how is)\b", re.I)


def wants_script(request: str) -> bool:
    if _QUESTION_START.match(request) and not re.search(r"\b(script|code)\b", request, re.I):
        return False
    return bool(_SCRIPT.search(request))


def bow_cosine(a: str, b: str) -> float:
    """Cosine similarity of word counts, 0 to 1. A stand-in until the
    embedding model is wired in; the threshold is pilot-set either way."""
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    dot = sum(ca[t] * cb[t] for t in ca)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Example:
    """A verified example or parameter template, described by the request
    it answers (assistant/examples/<name>.py holds the script)."""

    name: str
    request: str
    path: str = ""


@dataclass
class Decision:
    route: str
    reason: str
    example: Optional[Example] = None
    similarity: float = 0.0
    weak_match: bool = False


class Router:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        cache: Optional[ExactCache] = None,
        examples: Sequence[Example] = (),
        similarity: Callable[[str, str], float] = bow_cosine,
        close_match: float = config.CLOSE_MATCH_COSINE,
        weak_match_score: float = config.WEAK_MATCH_BM25,
        commit: str = "",
    ):
        self.retriever = retriever
        self.cache = cache
        self.examples = list(examples)
        self.similarity = similarity
        self.close_match = close_match
        self.weak_match_score = weak_match_score
        self.commit = commit

    def route(self, request: str) -> Decision:
        if self.cache is not None and self.cache.get(request, self.commit) is not None:
            return Decision(REPEAT, "exact repeat of a stored request")

        if wants_script(request):
            best, sim = None, 0.0
            for ex in self.examples:
                s = self.similarity(request, ex.request)
                if s > sim:
                    best, sim = ex, s
            if best is not None and sim >= self.close_match:
                return Decision(TEMPLATE, "close to verified example '%s'" % best.name, best, sim)
            why = "no verified example is close enough" if self.examples else "no verified examples yet"
            return Decision(NEW_SCRIPT, why, best, sim)

        weak = False
        if self.retriever is not None:
            weak = self.retriever.best_lexical_score(request) < self.weak_match_score
        reason = "question; weak match in the docs, use fallback model" if weak else "question"
        return Decision(QUESTION, reason, weak_match=weak)


def load_examples(folder) -> List[Example]:
    """Read verified examples: each .py file starts with a docstring whose
    first line is 'Request: <the request it answers>'."""
    import ast
    from pathlib import Path

    out = []
    for py in sorted(Path(folder).glob("*.py")):
        doc = ast.get_docstring(ast.parse(py.read_text(encoding="utf-8"))) or ""
        first = doc.splitlines()[0] if doc else ""
        if first.lower().startswith("request:"):
            out.append(Example(py.stem, first.split(":", 1)[1].strip(), str(py)))
    return out
