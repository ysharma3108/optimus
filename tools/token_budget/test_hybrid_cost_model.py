"""Checks for hybrid_cost_model.py.  Run: python -m unittest tools/token_budget/test_hybrid_cost_model.py"""
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(__file__))
import hybrid_cost_model as m  # noqa: E402


class PricesAndTokens(unittest.TestCase):
    def test_haiku_question_matches_hand_calculation(self):
        p = m.Params()
        q_in = 596 * 1.3 + 69.5 + 5 * (384 + 15) * 1.3
        out = 250 * 1.44 * 1.3
        usd = (q_in * 0.10 + out * 0.50) / 1e6
        got = m.route_cost(p, "Q_claude")["claude_pence"]
        self.assertAlmostEqual(got, usd * m.GBP_PER_USD * 100, places=9)

    def test_cache_read_is_cheaper_than_input_on_sonnet(self):
        pin, _, pwrite, pread = m.PRICES["sonnet-5.5"]
        self.assertEqual(pread, pin * 0.05)
        self.assertEqual(pwrite, pin * 1.25)

    def test_batch_halves_cost(self):
        p = replace(m.Params(), prompt_cache=False)
        full = m.route_cost(p, "S3")["claude_pence"]
        half = m.route_cost(p, "S3", batch=True)["claude_pence"]
        self.assertAlmostEqual(half, full / 2)


class Loop(unittest.TestCase):
    def test_stored_code_costs_nothing(self):
        c = m.route_cost(m.Params(), "S1")
        self.assertEqual(c["pence"], 0)
        self.assertEqual(c["claude_calls"], 0)

    def test_call_count_with_certain_pass(self):
        p = replace(m.Params(), p_first=1.0)
        self.assertAlmostEqual(m.route_cost(p, "S3")["claude_calls"], 1.0)

    def test_call_count_with_certain_fail_is_the_worst_case(self):
        # write + 2 fixes + restart write + 2 fixes = 6 Sonnet calls, all unsolved
        p = replace(m.Params(), p_first=0.0, p_fix1=0.0)
        c = m.route_cost(p, "S3")
        self.assertAlmostEqual(c["claude_calls"], 6.0)
        self.assertAlmostEqual(c["p_unsolved"], 1.0)

    def test_expected_calls_central(self):
        # P(reach call) = 1, .5, .5*.5, .5*.5*.8, then restart .5*.5*.8*.5 ...
        p = m.Params()
        probs = [c.prob for c in m.route_calls(p, "S3")[0]]
        self.assertAlmostEqual(probs[0], 1.0)
        self.assertAlmostEqual(probs[1], 0.5)
        self.assertAlmostEqual(probs[2], 0.25)
        self.assertAlmostEqual(probs[3], 0.25 * 0.8)

    def test_template_route_only_calls_claude_on_local_failure(self):
        p = replace(m.Params(), p_local_template=1.0)
        self.assertEqual(m.route_cost(p, "S2")["claude_calls"], 0)

    def test_caching_never_costs_more_on_multi_call_loops(self):
        p = m.Params()
        on = m.route_cost(p, "S3")["claude_pence"]
        off = m.route_cost(replace(p, prompt_cache=False), "S3")["claude_pence"]
        self.assertLess(on, off)


class Project(unittest.TestCase):
    def test_scenarios_are_ordered(self):
        totals = [m.project_total(replace(m.Params(), **m.SCENARIOS[s]))[1]
                  for s in ("optimistic", "central", "pessimistic")]
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])

    def test_waterfall_ends_at_the_project_total(self):
        p = m.Params()
        self.assertAlmostEqual(m.waterfall(p)[-1][1], m.project_total(p)[1])

    def test_pessimistic_fits_the_35_pound_budget(self):
        p = replace(m.Params(), **m.SCENARIOS["pessimistic"])
        self.assertLess(m.project_total(p)[1], 35)


if __name__ == "__main__":
    unittest.main()
