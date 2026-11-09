"""Command line for the OptimUS assistant.

Run from the repository root (works in PowerShell and bash):

    python -m assistant index                      # build the knowledge base
    python -m assistant search "how do I make a bowl source"
    python -m assistant route "write a script for a 500 kHz bowl in water"
    python -m assistant check my_script.py         # syntax, API names, mesh
    python -m assistant ask "what preconditioners are there" --model local
    python -m assistant log                        # token use so far

`ask` needs a model: --model local (Ollama running) or --model haiku /
--model sonnet (pip install anthropic, and an API key). Everything else
works offline with no extra packages.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config
from .checks import run_all
from .ingest import api_names, build_pieces, load_index, save_index
from .retrieve import Retriever
from .router import Router, load_examples
from .tokenlog import TokenLog

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".assistant"
INDEX = STATE / "index.json"
LOG = STATE / "token_log.jsonl"
EXAMPLES = ROOT / "assistant" / "examples"


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def _pieces():
    if INDEX.exists():
        return load_index(INDEX)
    return build_pieces(ROOT)


def cmd_index(a):
    pieces = build_pieces(ROOT)
    STATE.mkdir(exist_ok=True)
    save_index(pieces, INDEX, _commit())
    kinds = {}
    for p in pieces:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    print("indexed %d pieces %s -> %s" % (len(pieces), kinds, INDEX))


def cmd_search(a):
    for h in Retriever(_pieces()).search(a.query, a.k):
        print("%2d. %6.2f  %s" % (h.rank, h.score, h.piece.citation()))


def cmd_route(a):
    r = Router(Retriever(_pieces()), examples=load_examples(EXAMPLES) if EXAMPLES.exists() else ())
    d = r.route(a.request)
    print("route: %s\nreason: %s" % (d.route, d.reason))
    if d.example:
        print("closest example: %s (similarity %.2f)" % (d.example.name, d.similarity))


def cmd_check(a):
    code = Path(a.script).read_text(encoding="utf-8")
    issues = run_all(code, api_names(ROOT))
    for i in issues:
        print(i)
    if not issues:
        print("no issues found")
    sys.exit(1 if any(i.level == "error" for i in issues) else 0)


def _client(name, log):
    from .llm import AnthropicClient, OllamaClient

    if name == "local":
        return OllamaClient(log=log)
    model = {"haiku": config.MODEL_QUESTIONS_FALLBACK, "sonnet": config.MODEL_SCRIPTS}[name]
    return AnthropicClient(model, log=log)


def cmd_ask(a):
    from .answer import answer

    log = TokenLog(LOG)
    retriever = Retriever(_pieces())
    fallback = _client(a.fallback, log) if a.fallback else None
    weak = retriever.best_lexical_score(a.question) < config.WEAK_MATCH_BM25
    ans = answer(a.question, retriever, _client(a.model, log), fallback, k=a.k, weak_match=weak)
    print(ans.text)
    print("\nSources:")
    for s in ans.sources:
        print(" -", s)
    if ans.fell_back:
        print("(answered by fallback model: %s)" % ans.reason)


def cmd_log(a):
    if not LOG.exists():
        print("no model calls logged yet")
        return
    print(json.dumps(TokenLog(LOG).summary(), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m assistant", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=config.TOP_K)
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("route"); p.add_argument("request"); p.set_defaults(fn=cmd_route)
    p = sub.add_parser("check"); p.add_argument("script"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("ask"); p.add_argument("question")
    p.add_argument("--model", choices=["local", "haiku", "sonnet"], default="local")
    p.add_argument("--fallback", choices=["haiku", "sonnet"], default="haiku")
    p.add_argument("-k", type=int, default=config.TOP_K)
    p.set_defaults(fn=cmd_ask)
    sub.add_parser("log").set_defaults(fn=cmd_log)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
