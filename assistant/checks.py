"""Checks on a generated script that need no model and no solver run.

They run before the Docker run, because they take milliseconds and catch the
most common failures of generated code: syntax errors, made-up API names and
a mesh too coarse for the frequency.
"""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import config

ERROR, WARNING, INFO = "error", "warning", "info"

_DATA = Path(__file__).resolve().parent / "data" / "material_speeds.json"


@dataclass
class Issue:
    level: str  # "error", "warning" or "info"
    check: str
    message: str
    line: int = 0

    def __str__(self) -> str:
        where = " (line %d)" % self.line if self.line else ""
        return "[%s] %s%s: %s" % (self.level, self.check, where, self.message)


# ------------------------------------------------------------------ syntax


def check_syntax(code: str) -> List[Issue]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return [Issue(ERROR, "syntax", e.msg, e.lineno or 0)]
    return []


# ------------------------------------------------------------------ API names


def _dotted(node: ast.AST) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _aliases(tree: ast.AST, package: str) -> Dict[str, str]:
    """Local name -> full dotted name, for every import from the package."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == package or a.name.startswith(package + "."):
                    if a.asname:
                        out[a.asname] = a.name
                    else:
                        out[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == package or node.module.startswith(package + "."):
                for a in node.names:
                    if a.name != "*":
                        out[a.asname or a.name] = node.module + "." + a.name
    return out


def check_api_names(code: str, api: Dict[str, str], package: str = "optimus") -> List[Issue]:
    """Every `optimus.x.y` used in the script must exist in the source.

    `api` is ingest.api_names(): dotted name -> "module" / "function" /
    "class" / "name". A chain is checked up to the first part that is not a
    module, because attributes of objects cannot be checked statically.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # reported by check_syntax
    aliases = _aliases(tree, package)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # so only the outermost a.b.c chain is checked
    issues, seen = [], set()

    def check(full: str, line: int):
        parts = full.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            kind = api.get(name)
            if kind is None:
                if name not in seen:
                    seen.add(name)
                    issues.append(Issue(ERROR, "api-name", "'%s' does not exist in OptimUS" % name, line))
                return
            if kind != "module":
                return  # attributes of a function, class or object: not checkable

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == package or node.module.startswith(package + "."):
                check(node.module, node.lineno)
                for a in node.names:
                    if a.name != "*":
                        check(node.module + "." + a.name, node.lineno)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(package + "."):
                    check(a.name, node.lineno)
        elif isinstance(node, ast.Attribute) and not isinstance(getattr(node, "_parent", None), ast.Attribute):
            dotted = _dotted(node)
            if dotted:
                head, _, rest = dotted.partition(".")
                if head in aliases:
                    check(aliases[head] + ("." + rest if rest else ""), node.lineno)
    return issues


# ------------------------------------------------------------------ mesh size

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST, env: Dict[str, float]) -> Optional[float]:
    """Evaluate simple arithmetic on numbers and known names; else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left, env), _eval(node.right, env)
        if a is None or b is None:
            return None
        try:
            return _OPS[type(node.op)](a, b)
        except (ZeroDivisionError, OverflowError):
            return None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        a = _eval(node.operand, env)
        return None if a is None else _OPS[type(node.op)](a)
    return None


def material_speeds() -> Dict[str, float]:
    return json.loads(_DATA.read_text(encoding="utf-8"))["speed_of_sound_m_per_s"]


def elements_per_wavelength(frequency: float, element_size: float, speed_of_sound: float) -> float:
    """Wavelength / element size, with wavelength = c / f."""
    return speed_of_sound / frequency / element_size


def mesh_verdict(epw: float) -> str:
    if epw < config.ELEMENTS_PER_WAVELENGTH_FAIL:
        return ERROR
    if epw < config.ELEMENTS_PER_WAVELENGTH_WARN:
        return WARNING
    return INFO


def check_mesh(code: str, speeds: Optional[Dict[str, float]] = None) -> List[Issue]:
    """Elements per wavelength from the numbers written in the script.

    Finds the frequency (a variable called frequency/freq/f0, or a
    frequency= argument), every element_size= argument or variable, and
    every material (load_material("name") or a speed_of_sound= argument).
    The wavelength uses the slowest material, which gives the shortest
    wavelength. When a value cannot be read statically, the check says so
    as "info" rather than guessing.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    speeds = material_speeds() if speeds is None else speeds
    env: Dict[str, float] = {}
    freqs, sizes, cs = [], [], []
    unknown_sizes = 0

    # Visit nodes in source order so later uses see earlier assignments.
    nodes = sorted(
        (n for n in ast.walk(tree) if hasattr(n, "lineno")), key=lambda n: (n.lineno, n.col_offset)
    )
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = _eval(node.value, env)
            name = node.targets[0].id
            if v is not None:
                env[name] = v
                low = name.lower()
                if low in ("frequency", "freq", "f0"):
                    freqs.append((v, node.lineno))
                if low in ("element_size", "elementsize", "meshwidth", "mesh_size"):
                    sizes.append((v, node.lineno))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "frequency":
                    v = _eval(kw.value, env)
                    if v is not None:
                        freqs.append((v, node.lineno))
                elif kw.arg in ("element_size", "meshwidth"):
                    v = _eval(kw.value, env)
                    if v is not None:
                        sizes.append((v, node.lineno))
                    else:
                        unknown_sizes += 1
                elif kw.arg == "speed_of_sound":
                    v = _eval(kw.value, env)
                    if v is not None:
                        cs.append(v)
            fn = _dotted(node.func) or ""
            if fn.endswith("load_material") and node.args:
                arg = node.args[0]
                names = []
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names = [arg.value]
                elif isinstance(arg, (ast.List, ast.Tuple)):
                    names = [e.value for e in arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                for n in names:
                    if n.lower() in speeds:
                        cs.append(speeds[n.lower()])

    if not freqs:
        return [Issue(INFO, "mesh", "no frequency found as a number; mesh not checked")]
    if not sizes:
        msg = "no element size found as a number; mesh not checked"
        if unknown_sizes:
            msg = "element size is computed at run time; mesh not checked statically"
        return [Issue(INFO, "mesh", msg)]

    c = min(cs) if cs else config.DEFAULT_SPEED_OF_SOUND
    f = max(v for v, _ in freqs)  # highest frequency = shortest wavelength
    issues = []
    first_line = {}
    for h, line in sizes:  # the same value set once and passed on: report it once
        first_line.setdefault(h, line)
    for h, line in sorted(first_line.items(), key=lambda kv: kv[1]):
        if h <= 0:
            issues.append(Issue(ERROR, "mesh", "element size must be positive", line))
            continue
        epw = elements_per_wavelength(f, h, c)
        level = mesh_verdict(epw)
        issues.append(
            Issue(
                level,
                "mesh",
                "%.1f elements per wavelength (f = %g Hz, c = %g m/s, element size = %g m); "
                "rule: at least %g, ideally %g"
                % (epw, f, c, h, config.ELEMENTS_PER_WAVELENGTH_FAIL, config.ELEMENTS_PER_WAVELENGTH_WARN),
                line,
            )
        )
    return issues


# ------------------------------------------------------------------ all checks


def run_all(code: str, api: Optional[Dict[str, str]] = None) -> List[Issue]:
    issues = check_syntax(code)
    if issues:
        return issues
    if api is not None:
        issues += check_api_names(code, api)
    issues += check_mesh(code)
    return issues


def has_errors(issues: List[Issue]) -> bool:
    return any(i.level == ERROR for i in issues)
