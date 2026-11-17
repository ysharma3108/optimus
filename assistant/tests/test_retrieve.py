import json

import pytest

from assistant.ingest import Piece, build_pieces
from assistant.metrics import hit_rate_sweep
from assistant.retrieve import BM25, Hit, Retriever, DenseIndex, cosine, rrf, tokenize


def piece(i, text):
    return Piece(id=str(i), text=text, source="s%d" % i, kind="doc", location="whole file")


def test_tokenize_keeps_identifiers_and_parts():
    toks = tokenize("Use create_bowl() at 500e3 Hz")
    assert "create_bowl" in toks and "create" in toks and "bowl" in toks
    assert "use" in toks and "the" not in tokenize("the")


def test_bm25_ranks_the_matching_piece_first():
    pieces = [piece(0, "plane wave incident field"), piece(1, "bowl transducer source create_bowl"),
              piece(2, "material database water bone")]
    hits = BM25(pieces).search("how to create a bowl source")
    assert hits[0].piece.id == "1"
    assert all(h.score > 0 for h in hits)


def test_bm25_idf_stays_positive_for_common_terms():
    pieces = [piece(i, "optimus common term") for i in range(5)]
    assert all(v > 0 for v in BM25(pieces).idf.values())


def test_rrf_follows_the_formula():
    a, b = piece(0, "a"), piece(1, "b")
    fused = rrf([[Hit(a, 1, 1), Hit(b, 1, 2)], [Hit(b, 1, 1)]], k=60)
    assert fused[0].piece.id == "1"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)


def test_dense_ranking_is_fused_when_given():
    pieces = [piece(0, "alpha"), piece(1, "beta")]
    fake_embed = lambda texts: [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]
    r = Retriever(pieces, DenseIndex(pieces, fake_embed))
    assert r.search("alpha", 2)[0].piece.id == "0"
    assert cosine([1, 0], [0, 1]) == 0


def test_baseline_hit_rate_on_seed_questions(repo):
    """Regression guard for the search baseline, NOT a success criterion.

    Measured on 9 Nov 2026 with BM25 alone: hit@8 = 8/12, hit@20 = 12/12.
    The real target (R1) is set from the pilot question set.
    """
    r = Retriever(build_pieces(repo))
    rows = [json.loads(l) for l in (repo / "assistant/eval/seed_questions.jsonl").read_text().splitlines() if l]
    results = [([h.piece.source for h in r.search(q["question"], 20)], q["relevant"]) for q in rows]
    sweep = hit_rate_sweep(results, (8, 20))
    assert sweep[8]["hits"] >= 8
    assert sweep[20]["hits"] == len(rows)
