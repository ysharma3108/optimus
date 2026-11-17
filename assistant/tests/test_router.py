from assistant.cache import ExactCache
from assistant.ingest import Piece
from assistant.retrieve import Retriever
from assistant.router import (NEW_SCRIPT, QUESTION, REPEAT, TEMPLATE, Example, Router, bow_cosine,
                              load_examples, wants_script)


def test_script_requests_are_recognised():
    assert wants_script("Write a script for a 500 kHz bowl transducer in water")
    assert wants_script("Can you give me code to simulate a piston source?")
    assert wants_script("Simulate a plane wave hitting a bone sphere")
    assert not wants_script("What does the OSRC preconditioner do?")
    assert not wants_script("How does the BEM handle nested domains?")


def test_repeat_route_uses_the_cache(tmp_path):
    cache = ExactCache(tmp_path / "c.json")
    cache.put("What is OptimUS?", "A BEM library.", "answer", commit="abc")
    r = Router(cache=cache, commit="abc")
    assert r.route("  what is optimus ").route == REPEAT
    assert Router(cache=cache, commit="def").route("What is OptimUS?").route == QUESTION  # stale


def test_template_vs_new_script():
    ex = Example("bowl_500k", "write a script for a bowl transducer at 500 kHz in water")
    r = Router(examples=[ex], close_match=0.6)
    d = r.route("Write a script for a bowl transducer at 250 kHz in water")
    assert d.route == TEMPLATE and d.example is ex and d.similarity >= 0.6
    assert r.route("Write code for a nested skull model with three layers").route == NEW_SCRIPT


def test_weak_match_flags_fallback():
    r = Router(Retriever([Piece("1", "bowl source", "s", "doc", "x")]), weak_match_score=0.5)
    assert r.route("What is a bowl source?").weak_match is False
    assert r.route("What is the capital of France?").weak_match is True


def test_bow_cosine_bounds():
    assert bow_cosine("bowl source", "bowl source") == __import__("pytest").approx(1.0)
    assert bow_cosine("bowl", "piston") == 0.0


def test_load_examples(tmp_path):
    (tmp_path / "bowl.py").write_text('"""Request: bowl at 500 kHz in water\n\nVerified 2026-11-09."""\nx = 1\n')
    (tmp_path / "other.py").write_text('"""No header."""\n')
    exs = load_examples(tmp_path)
    assert [e.name for e in exs] == ["bowl"] and exs[0].request == "bowl at 500 kHz in water"
