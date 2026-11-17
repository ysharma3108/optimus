"""Route 3: write, check, run, fix; stop early; one fresh restart.

The shape follows the debugging-loop research (2-3 Nov 2026):
  - plan before code (AlphaCodium, Ridnik et al. 2024),
  - fix using the real error message (Self-Debugging, Chen et al. 2024),
  - at most config.MAX_FIXES fixes, because fix ability decays fast
    (Debugging Decay Index, Adnan & Kuhn 2025),
  - then one fresh restart with the history cleared and a one-line lesson
    (Reflexion, Shinn et al. 2023), again with at most MAX_FIXES fixes,
  - then hand over to the user with the last error.
Worst case: 1 + MAX_FIXES + 1 + MAX_FIXES = 6 model calls.

Model calls, checks and the Docker run are passed in as functions, so the
loop is tested without any model or Docker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import config
from .checks import Issue, has_errors


@dataclass
class RunResult:
    ok: bool
    output: str = ""
    error: str = ""
    seconds: float = 0.0


@dataclass
class Attempt:
    stage: str  # "write", "fix", "restart"
    code: str
    passed_checks: bool
    ran: bool
    ok: bool
    error: str = ""


@dataclass
class LoopResult:
    ok: bool
    code: str
    attempts: List[Attempt] = field(default_factory=list)
    handed_over: bool = False

    @property
    def model_calls(self) -> int:
        return len(self.attempts)


Write = Callable[[str, str], str]  # (request, lesson) -> code
Fix = Callable[[str, str, str], str]  # (request, code, error) -> code
Check = Callable[[str], List[Issue]]
Run = Callable[[str], RunResult]


def _try(code: str, check: Check, run: Run, stage: str) -> Attempt:
    issues = check(code)
    if has_errors(issues):
        msg = "\n".join(str(i) for i in issues if i.level == "error")
        return Attempt(stage, code, passed_checks=False, ran=False, ok=False, error=msg)
    result = run(code)
    return Attempt(stage, code, passed_checks=True, ran=True, ok=result.ok, error=result.error)


def lesson_from(attempts: List[Attempt]) -> str:
    """A one-line summary of what went wrong, carried into the restart
    instead of the full failed history."""
    last = attempts[-1] if attempts else None
    if last is None or not last.error:
        return ""
    first_line = last.error.strip().splitlines()[-1][:200]
    return "A previous attempt failed with: " + first_line + ". Avoid that approach."


def solve(
    request: str,
    write: Write,
    fix: Fix,
    check: Check,
    run: Run,
    max_fixes: int = config.MAX_FIXES,
    max_restarts: int = config.MAX_FRESH_RESTARTS,
) -> LoopResult:
    attempts: List[Attempt] = []
    lesson = ""
    for round_ in range(1 + max_restarts):
        code = write(request, lesson)
        a = _try(code, check, run, "write" if round_ == 0 else "restart")
        attempts.append(a)
        if a.ok:
            return LoopResult(True, code, attempts)
        for _ in range(max_fixes):
            code = fix(request, code, a.error)
            a = _try(code, check, run, "fix")
            attempts.append(a)
            if a.ok:
                return LoopResult(True, code, attempts)
        lesson = lesson_from(attempts)
    return LoopResult(False, attempts[-1].code, attempts, handed_over=True)
