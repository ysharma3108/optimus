from assistant import config
from assistant.checks import Issue
from assistant.loop import RunResult, lesson_from, solve


def no_issues(code):
    return []


def run_ok_if(marker):
    return lambda code: RunResult(marker in code, error="" if marker in code else "Traceback\nValueError: bad mesh")


def test_passes_first_time():
    res = solve("req", lambda r, l: "GOOD", lambda r, c, e: c, no_issues, run_ok_if("GOOD"))
    assert res.ok and res.model_calls == 1 and res.attempts[0].stage == "write"


def test_fix_uses_the_error_message():
    seen = []

    def fix(request, code, error):
        seen.append(error)
        return "GOOD"

    res = solve("req", lambda r, l: "bad", fix, no_issues, run_ok_if("GOOD"))
    assert res.ok and [a.stage for a in res.attempts] == ["write", "fix"]
    assert "ValueError: bad mesh" in seen[0]


def test_gives_up_after_the_cited_cap_and_one_restart():
    lessons = []

    def write(request, lesson):
        lessons.append(lesson)
        return "bad"

    res = solve("req", write, lambda r, c, e: "bad", no_issues, run_ok_if("GOOD"))
    assert not res.ok and res.handed_over
    expected = (1 + config.MAX_FIXES) * (1 + config.MAX_FRESH_RESTARTS)
    assert res.model_calls == expected == 6
    assert [a.stage for a in res.attempts] == ["write", "fix", "fix", "restart", "fix", "fix"]
    assert lessons[0] == "" and "bad mesh" in lessons[1]


def test_failed_checks_skip_the_run():
    runs = []

    def run(code):
        runs.append(code)
        return RunResult(True)

    check = lambda code: [Issue("error", "api-name", "nope")] if "bad" in code else []
    res = solve("req", lambda r, l: "bad", lambda r, c, e: "fine", check, run)
    assert res.ok and runs == ["fine"]
    assert res.attempts[0].ran is False and "nope" in res.attempts[0].error


def test_lesson_is_one_line():
    assert lesson_from([]) == ""
