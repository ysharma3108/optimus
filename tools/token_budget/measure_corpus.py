"""Measure how many tokens the OptimUS knowledge base really contains.

Reads every notebook in notebooks/, keeps the markdown and code cells
(outputs and images are dropped, as a RAG index would drop them), cuts
them into retrieval chunks at several target sizes, and counts tokens.

Usage (Python 3.8+):
    pip install tokenizers
    python tools/token_budget/measure_corpus.py --tokenizer path/to/tokenizer.json

The tokenizer file used for the reported numbers is the Claude tokenizer
that shipped inside the anthropic Python SDK up to version 0.34
(anthropic/tokenizer.json). Anthropic says newer Claude models use a
different tokenizer, so treat these counts as a measured baseline and
apply the uplift range given in the cost model.
"""

import argparse
import glob
import json
import os
import statistics

from tokenizers import Tokenizer


def load_cells(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") not in ("markdown", "code"):
            continue
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = src.strip()
        if not src:
            continue
        if c["cell_type"] == "code":
            src = "```python\n" + src + "\n```"
        cells.append(src)
    return cells


def chunk_cells(cells, count, target):
    """Greedy cell-aware chunking: join whole cells until the target is
    reached; a single cell bigger than the target is split by lines."""
    chunks, cur, cur_n = [], [], 0
    for cell in cells:
        n = count(cell)
        if n > target:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_n = [], 0
            part, part_n = [], 0
            for line in cell.splitlines():
                ln = count(line) + 1
                if part and part_n + ln > target:
                    chunks.append("\n".join(part))
                    part, part_n = [], 0
                part.append(line)
                part_n += ln
            if part:
                chunks.append("\n".join(part))
            continue
        if cur and cur_n + n > target:
            chunks.append("\n\n".join(cur))
            cur, cur_n = [], 0
        cur.append(cell)
        cur_n += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--notebooks", default="notebooks")
    ap.add_argument("--source", default="optimus")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra text files to count (scripts, prompts)")
    ap.add_argument("--json", help="write results to this JSON file")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    def count(text):
        return len(tok.encode(text).ids)

    out = {"notebooks": {}, "chunking": {}, "extra": {}}
    all_cells = []
    for path in sorted(glob.glob(os.path.join(args.notebooks, "*.ipynb"))):
        cells = load_cells(path)
        all_cells.append(cells)
        text = "\n\n".join(cells)
        out["notebooks"][os.path.basename(path)] = {
            "cells": len(cells),
            "words": len(text.split()),
            "chars": len(text),
            "tokens": count(text),
        }

    nbs = out["notebooks"].values()
    tot_tokens = sum(v["tokens"] for v in nbs)
    tot_words = sum(v["words"] for v in nbs)
    tot_chars = sum(v["chars"] for v in nbs)
    cell_sizes = [count(c) for cells in all_cells for c in cells]
    out["totals"] = {
        "notebooks": len(out["notebooks"]),
        "cells": len(cell_sizes),
        "tokens": tot_tokens,
        "words": tot_words,
        "chars": tot_chars,
        "tokens_per_word": round(tot_tokens / tot_words, 2),
        "chars_per_token": round(tot_chars / tot_tokens, 2),
        "cell_tokens_median": statistics.median(cell_sizes),
        "cell_tokens_mean": round(statistics.mean(cell_sizes), 1),
        "cell_tokens_p90": sorted(cell_sizes)[int(0.9 * len(cell_sizes))],
        "cell_tokens_max": max(cell_sizes),
    }

    for target in (256, 512, 1024):
        sizes = [count(ch) for cells in all_cells
                 for ch in chunk_cells(cells, count, target)]
        out["chunking"][target] = {
            "chunks": len(sizes),
            "mean": round(statistics.mean(sizes), 1),
            "median": statistics.median(sizes),
            "p90": sorted(sizes)[int(0.9 * len(sizes))],
            "max": max(sizes),
        }

    src_files = glob.glob(os.path.join(args.source, "**", "*.py"), recursive=True)
    src_text = ""
    for p in src_files:
        with open(p, encoding="utf-8") as f:
            src_text += f.read() + "\n"
    out["source_code"] = {"files": len(src_files), "tokens": count(src_text)}

    for p in args.extra:
        with open(p, encoding="utf-8") as f:
            t = f.read()
        out["extra"][os.path.basename(p)] = {"lines": t.count("\n"), "tokens": count(t)}

    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
