"""Build the knowledge base: notebooks, OptimUS source and docs cut into pieces.

Every piece keeps where it came from (file, cell or line range, heading) so
that an answer can cite it and a reader can check it.

Only the standard library is used, so this runs on any Python 3.8+ without
OptimUS or bempp installed.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from . import config


@dataclass
class Piece:
    """One retrievable piece of the knowledge base."""

    id: str
    text: str
    source: str  # path relative to the repo root
    kind: str  # "notebook", "code", "doc" or "example"
    location: str  # "cells 3-5", "lines 14-60", ...
    heading: str = ""
    symbols: List[str] = field(default_factory=list)  # API names defined here

    def citation(self) -> str:
        where = self.source + ", " + self.location
        return where + (" (" + self.heading + ")" if self.heading else "")

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Rough token count, used only to size pieces (see config.py)."""
    return int(len(text) / config.CHARS_PER_TOKEN_LEGACY * config.TOKENIZER_UPLIFT) + 1


# ------------------------------------------------------------------ notebooks

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.strip()


def notebook_cells(path: Path) -> List[dict]:
    """Markdown and code cells of a notebook, with outputs dropped.

    Outputs are dropped because they are long (images, tables) and are a
    result of running the code, not documentation of how to use it.
    """
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for index, cell in enumerate(nb.get("cells", [])):
        kind = cell.get("cell_type")
        if kind not in ("markdown", "code"):
            continue
        src = _cell_source(cell)
        if not src:
            continue
        if kind == "code":
            src = "```python\n" + src + "\n```"
        cells.append({"index": index, "kind": kind, "text": src})
    return cells


def _last_heading(text: str, current: str) -> str:
    found = _HEADING.findall(text)
    return found[-1][1].strip() if found else current


def chunk_notebook(
    path: Path,
    root: Path,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    count: Callable[[str], int] = estimate_tokens,
) -> List[Piece]:
    """Cell-aware chunking: whole cells are joined until the target size.

    A new markdown heading also starts a new piece, so a piece stays on one
    topic. A single cell larger than the target becomes its own piece
    rather than being cut mid-code, so a code example is never split.
    """
    rel = path.relative_to(root).as_posix()
    title = path.stem
    pieces: List[Piece] = []
    buf: List[dict] = []
    buf_tokens = 0
    heading = ""
    buf_heading = ""

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        first, last = buf[0]["index"], buf[-1]["index"]
        loc = "cell %d" % first if first == last else "cells %d-%d" % (first, last)
        text = "# " + title + "\n\n" + "\n\n".join(c["text"] for c in buf)
        pieces.append(
            Piece(
                id="%s#%d" % (rel, first),
                text=text,
                source=rel,
                kind="notebook",
                location=loc,
                heading=buf_heading,
            )
        )
        buf, buf_tokens = [], 0

    for cell in notebook_cells(path):
        n = count(cell["text"])
        starts_section = cell["kind"] == "markdown" and _HEADING.search(cell["text"])
        if buf and (starts_section or buf_tokens + n > target_tokens):
            flush()
        if not buf:
            buf_heading = heading
        if cell["kind"] == "markdown":
            heading = _last_heading(cell["text"], heading)
            if not buf:
                buf_heading = heading
        buf.append(cell)
        buf_tokens += n
    flush()
    return pieces


# ------------------------------------------------------------------ source code


def _public(name: str) -> bool:
    return not name.startswith("_")


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def chunk_source(path: Path, root: Path) -> List[Piece]:
    """One piece per public top-level function or class, with its docstring
    and signature. Classes include their public method signatures."""
    rel = path.relative_to(root).as_posix()
    mod = module_name(path, root)
    text = path.read_text(encoding="utf-8")
    try:
        tree = _parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    pieces = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or not _public(node.name):
            continue
        start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
        if isinstance(node, ast.FunctionDef):
            header = "def " + node.name + "(" + ast.unparse(node.args) + ")"
            doc = ast.get_docstring(node) or ""
            body = header + "\n\n" + doc
        else:
            doc = ast.get_docstring(node) or ""
            methods = [
                "    def " + m.name + "(" + ast.unparse(m.args) + ")"
                for m in node.body
                if isinstance(m, ast.FunctionDef) and (_public(m.name) or m.name == "__init__")
            ]
            body = "class " + node.name + "\n\n" + doc + "\n\n" + "\n".join(methods)
        qual = mod + "." + node.name
        pieces.append(
            Piece(
                id="%s:%d" % (rel, start),
                text="# " + qual + "\n\n" + body.strip() + "\n\n" + _snippet(lines, start, end),
                source=rel,
                kind="code",
                location="lines %d-%d" % (start, end),
                heading=qual,
                symbols=[qual],
            )
        )
    return pieces


def _parse(text: str) -> ast.AST:
    """ast.parse without SyntaxWarnings from old-style escapes in OptimUS
    docstrings (for example "\\c" in LaTeX), which are harmless here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


def _snippet(lines: List[str], start: int, end: int, max_lines: int = 40) -> str:
    chunk = lines[start - 1 : min(end, start - 1 + max_lines)]
    more = "\n    # ... (%d more lines)" % (end - start + 1 - max_lines) if end - start + 1 > max_lines else ""
    return "```python\n" + "\n".join(chunk) + more + "\n```"


# ------------------------------------------------------------------ docs


def chunk_doc(path: Path, root: Path) -> List[Piece]:
    """A whole markdown or reStructuredText file as one piece.

    The repository's docs are short (README.md and docs/index.rst), so they
    are not split. Revisit if longer docs are added.
    """
    rel = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [Piece(id=rel + "#0", text=text, source=rel, kind="doc", location="whole file")]


# ------------------------------------------------------------------ everything


def build_pieces(
    root: Path,
    notebooks: str = "notebooks",
    package: str = "optimus",
    docs: Optional[Iterable[str]] = ("README.md", "docs/index.rst"),
    examples: str = "assistant/examples",
) -> List[Piece]:
    """All pieces for the repository at `root`, in a stable order."""
    root = Path(root)
    pieces: List[Piece] = []
    for nb in sorted((root / notebooks).glob("*.ipynb")):
        pieces.extend(chunk_notebook(nb, root))
    for py in sorted((root / package).rglob("*.py")):
        pieces.extend(chunk_source(py, root))
    for doc in docs or ():
        p = root / doc
        if p.exists():
            pieces.extend(chunk_doc(p, root))
    ex_dir = root / examples
    if ex_dir.exists():
        for py in sorted(ex_dir.rglob("*.py")):
            text = py.read_text(encoding="utf-8")
            rel = py.relative_to(root).as_posix()
            pieces.append(
                Piece(id=rel + "#0", text=text, source=rel, kind="example", location="whole file")
            )
    return pieces


def api_names(root: Path, package: str = "optimus") -> dict:
    """Every public dotted name the package defines, mapped to its kind.

    Kinds are "module", "function", "class" or "name" (a module-level
    variable or a re-exported import). Both the defining module
    (optimus.source.bowl.create_bowl) and the package-level names re-exported
    by `from .x import y` in __init__.py (optimus.source.create_bowl) are
    included. Used by checks.check_api_names to catch made-up API calls.
    """
    root = Path(root)
    names = {}
    for py in sorted((root / package).rglob("*.py")):
        names[module_name(py, root)] = "module"
    for py in sorted((root / package).rglob("*.py")):
        mod = module_name(py, root)
        try:
            tree = _parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _public(node.name):
                names[mod + "." + node.name] = "function"
            elif isinstance(node, ast.ClassDef) and _public(node.name):
                names[mod + "." + node.name] = "class"
            elif isinstance(node, ast.ImportFrom) and node.level >= 1:
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name != "*" and _public(local):
                        names.setdefault(mod + "." + local, "name")
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and _public(t.id):
                        names.setdefault(mod + "." + t.id, "name")
    return names


def save_index(pieces: List[Piece], path: Path, commit: str = "") -> None:
    data = {"commit": commit, "pieces": [p.to_dict() for p in pieces]}
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_index(path: Path) -> List[Piece]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Piece(**d) for d in data["pieces"]]
