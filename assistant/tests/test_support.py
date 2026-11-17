import pytest

from assistant.answer import NOT_COVERED, answer, citations, valid_citations
from assistant.cache import ExactCache, normalise
from assistant.ingest import Piece
from assistant.llm import FakeClient
from assistant.metrics import best_k, hit_rate_sweep, mcnemar_exact, wilson_interval
from assistant.retrieve import Retriever
from assistant.runner import docker_command, extract_error
from assistant.tokenlog import CallRecord, TokenLog, cost_usd, read_log


# ---------------------------------------------------------------- metrics
def test_wilson_interval_known_value():
    lo, hi = wilson_interval(8, 10)  # textbook value: 0.490 to 0.943
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_best_k_prefers_fewer_pieces_within_noise():
    results = [(["a"] * 20, ["a"])] * 9 + [(["x"] * 7 + ["a"], ["a"])]
    sweep = hit_rate_sweep(results, (1, 3, 8))
    assert sweep[8]["hits"] == 10 and sweep[1]["hits"] == 9
    assert best_k(sweep) == 1


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


# ---------------------------------------------------------------- cache
def test_cache_roundtrip(tmp_path):
    c = ExactCache(tmp_path / "c.json")
    c.put("How do I X?", "Like this.", "answer")
    assert ExactCache(tmp_path / "c.json").get("how do i x")["answer"] == "Like this."
    assert normalise("  A   b? ") == "a b"


# ---------------------------------------------------------------- token log
def test_token_log_writes_and_summarises(tmp_path):
    log = TokenLog(tmp_path / "log.jsonl")
    log.add(CallRecord("m", "question", 100, 10, seconds=1.0))
    log.add(CallRecord("m", "question", 300, 30, cache_read_tokens=50, seconds=3.0))
    s = TokenLog(tmp_path / "log.jsonl").summary()["question | m"]
    assert s["calls"] == 2 and s["mean_input"] == 200 and s["total_cache_read"] == 50
    assert len(list(read_log(tmp_path / "log.jsonl"))) == 2


def test_cost_from_a_given_price_table():
    prices = {"m": {"input": 2.0, "output": 10.0, "cache_read": 0.1}}
    rec = CallRecord("m", "q", 1_000_000, 100_000, cache_read_tokens=1_000_000)
    assert cost_usd(rec, prices) == pytest.approx(2.0 + 1.0 + 0.1)


def test_client_logs_every_call(tmp_path):
    log = TokenLog()
    FakeClient(["hi there"], log=log).complete("sys", [{"role": "user", "content": "q"}], route="question")
    assert log.records[0].route == "question" and log.records[0].output_tokens == 2


# ---------------------------------------------------------------- answers
PIECES = [Piece("1", "create_bowl makes a bowl source", "optimus/source/bowl.py", "code", "lines 14-62"),
          Piece("2", "plane wave field", "notebooks/Basic model.ipynb", "notebook", "cell 3")]


def test_citations_are_parsed_and_checked():
    assert citations("Use it [P1]. Also [P2][P1].") == [1, 2]
    assert valid_citations("ok [P1]", 2) and not valid_citations("ok [P3]", 2) and not valid_citations("ok", 2)


def test_good_local_answer_is_kept():
    local = FakeClient(["Use create_bowl [P1]."], model="local")
    haiku = FakeClient(["should not be called"], model="haiku")
    a = answer("how to make a bowl", Retriever(PIECES), local, haiku, k=2)
    assert a.model == "local" and not a.fell_back and a.sources == ["optimus/source/bowl.py, lines 14-62"]
    assert haiku.calls == []


def test_uncited_answer_falls_back():
    local = FakeClient(["Just call make_bowl()."], model="local")
    haiku = FakeClient(["Use create_bowl [P1]."], model="haiku")
    a = answer("how to make a bowl", Retriever(PIECES), local, haiku, k=2)
    assert a.fell_back and a.model == "haiku" and "citation" in a.reason


def test_not_covered_is_accepted_and_weak_match_goes_straight_to_fallback():
    local = FakeClient(["The OptimUS documentation " + NOT_COVERED + "."], model="local")
    assert not answer("bowl?", Retriever(PIECES), local, None).fell_back
    haiku = FakeClient(["x [P1]"], model="haiku")
    local2 = FakeClient([], model="local")
    a = answer("bowl?", Retriever(PIECES), local2, haiku, weak_match=True)
    assert a.fell_back and local2.calls == []


# ---------------------------------------------------------------- runner
def test_docker_command_is_isolated():
    cmd = docker_command("/tmp/run", "img", data_dir="/data")
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "/data:/work/Data:ro" in cmd and cmd[-3:] == ["img", "python3", "script.py"]


def test_extract_error_keeps_last_traceback():
    err = "noise\nTraceback (most recent call last):\n  old\nKeyError: 1\nTraceback (most recent call last):\n  File x\nValueError: mesh"
    out = extract_error(err)
    assert out.startswith("Traceback") and "KeyError" not in out and out.endswith("ValueError: mesh")
