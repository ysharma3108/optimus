"""Route 1: answer a question from the knowledge base, with citations.

The model sees the top pieces labelled [P1], [P2], ... and must cite them.
An answer is only accepted if it cites at least one piece and every label
it uses exists (citation checks in the spirit of ALCE, Gao et al. 2023).
If the local model's answer fails that, or the search match is weak, the
fallback model answers instead (hybrid design of 3 Nov 2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .llm import Client
from .retrieve import Hit, Retriever

SYSTEM = """You answer questions about OptimUS, a Python library for acoustic \
simulation with the boundary element method.
Use ONLY the numbered pieces of documentation given. After every sentence that \
uses a piece, cite it like [P2]. If the pieces do not contain the answer, say \
"The OptimUS documentation does not cover this." and stop. Do not invent \
function names or parameters. Keep answers short and give a code snippet only \
if the pieces show one."""

_CITE = re.compile(r"\[P(\d+)\]")


def build_user_message(question: str, hits: List[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append("[P%d] %s\n%s" % (i, h.piece.citation(), h.piece.text))
    return "Documentation pieces:\n\n" + "\n\n---\n\n".join(parts) + "\n\nQuestion: " + question


def citations(text: str) -> List[int]:
    return sorted({int(n) for n in _CITE.findall(text)})


def valid_citations(text: str, n_pieces: int) -> bool:
    cited = citations(text)
    return bool(cited) and all(1 <= c <= n_pieces for c in cited)


NOT_COVERED = "does not cover this"


@dataclass
class Answer:
    text: str
    model: str
    sources: List[str] = field(default_factory=list)
    fell_back: bool = False
    reason: str = ""


def answer(question: str, retriever: Retriever, primary: Client, fallback: Optional[Client] = None,
           k: int = config.TOP_K, weak_match: bool = False, request_id: str = "") -> Answer:
    hits = retriever.search(question, k)
    msg = [{"role": "user", "content": build_user_message(question, hits)}]

    def accept(text: str) -> bool:
        return NOT_COVERED in text or valid_citations(text, len(hits))

    def sources(text: str) -> List[str]:
        return [hits[c - 1].piece.citation() for c in citations(text) if 1 <= c <= len(hits)]

    reason = ""
    if not (weak_match and fallback is not None):
        reply = primary.complete(SYSTEM, msg, route="question", request_id=request_id)
        if accept(reply.text) or fallback is None:
            return Answer(reply.text, reply.model, sources(reply.text))
        reason = "no valid citation from the first model"
    else:
        reason = "weak search match"
    reply = fallback.complete(SYSTEM, msg, route="question_fallback", request_id=request_id)
    return Answer(reply.text, reply.model, sources(reply.text), fell_back=True, reason=reason)
