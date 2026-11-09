"""Cost model for the hybrid local/Claude OptimUS assistant (design of 3 Nov 2026).

Routes modelled (one per request type):
  Q  general question      -> local LLM answers from docs; Haiku 5.5 re-answers
                              when the local answer cites nothing or the match is weak
  S1 same values as stored -> return stored/notebook code. No model, no test.
  S2 same concept, new values -> local LLM fills a parameter template, script is run
                              and checked; on failure Sonnet 5.5 fixes it
  S3 no close match        -> Sonnet 5.5 writes new code, it is run, at most 2 fixes,
                              then one fresh restart (history cleared), then hand over

Every input carries a tag:
  [M] measured in this repo (measure_corpus.py, measured_corpus.json)
  [C] cited (paper, official price list); the source is in the comment
  [D] by design (a choice we make and can defend, e.g. an output length cap)
  [P] assumption the pilot run replaces with logged response.usage numbers

    python tools/token_budget/hybrid_cost_model.py            # tables
    python tools/token_budget/hybrid_cost_model.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace

# --------------------------------------------------------------------- prices
# [C] USD per million tokens, platform.claude.com/docs/en/about-claude/pricing,
# read 9 Nov 2026. Haiku 5.5 prices are for prompts up to 100K tokens; every
# prompt here is under 10K. Cache: 5-minute write = 1.25x input; read = 0.05x
# input on Sonnet 5.5, 0.1x on Haiku 5.5. Batch API = 50% off, stacks with cache.
PRICES = {
    #               input  output  cache_write_5m  cache_read
    "haiku-5.5":   (0.10,  0.50,   0.125,          0.01),
    "sonnet-5.5":  (2.00, 10.00,   2.50,           0.10),
    "opus-5.5":    (4.00, 20.00,   5.00,           0.20),
}
BATCH_DISCOUNT = 0.50
# [C] Bank of England spot, 23 Oct 2026: GBP 1 = USD 1.3387
GBP_PER_USD = 1 / 1.3387

# [C] Ofgem price cap, 1 Oct - 31 Dec 2026, electricity unit rate 26.32 p/kWh
PENCE_PER_KWH = 26.32


@dataclass(frozen=True)
class Params:
    # ---- tokenizer
    # [C] Anthropic token-counting docs: models from Opus 4.7 on (Haiku 5.5 and
    # Sonnet 5.5 included) use a tokenizer that produces about 1.0-1.35x the
    # tokens of the old one. Our [M] counts used the old tokenizer, so x1.3.
    uplift: float = 1.30

    # ---- request text (already real-text token counts, no uplift)
    # [C] mean user prompt, LMSYS-Chat-1M (Zheng et al., ICLR 2024, Table 1), Llama-2 tokens
    question: float = 69.5
    # [C] mean user prompt, WildChat (Zhao et al., ICLR 2024, Table 1), Llama-2 tokens
    script_request: float = 295.6

    # ---- measured in the repo (old tokenizer, uplift applied below)
    system_prompt_old: float = 596      # [M] example_system_prompt.md
    chunk_old: float = 384              # [M] mean piece, cell-aware chunking, 512 target
    chunk_label_old: float = 15         # [M] "[Notebook name #n]" tag
    script_old: float = 849             # [M] mean of the 3 starter scripts (655-1,110)
    words_tokens_old: float = 1.44      # [M] tokens per word of notebook prose

    # ---- retrieval
    # [C] k = 5: CodeRAG-Bench (Wang et al., NAACL 2025 Findings) "adding five
    # documents yields the best results in most settings"; Xu et al. (ICLR 2024)
    # top-5 beat top-10 and top-20 for both 4K and 32K models.
    k: int = 5

    # ---- outputs
    answer_words: float = 250           # [D] answer length cap in the system prompt
    script_notes_old: float = 100       # [D] five lines of explanation after a script
    thinking_question: float = 0        # [D] Haiku 5.5 with thinking "disabled" (allowed at effort <= high)
    thinking_script: float = 0          # [D] Sonnet 5.5 thinking {"type": "between_tools"}
    error_text: float = 600             # [P] traceback + checker message fed back on a failed run

    # ---- pass rates
    # [C anchor, P value] first-try pass. BigCodeBench (Zhuo et al., ICLR 2025):
    # best model 61.1% (Complete) / 51.1% (Instruct, natural-language request) vs
    # 97% for humans. OptimUS is a niche library, so 0.5 central.
    p_first: float = 0.5
    # [C anchor, P value] first fix pass rate. Olausson et al. (ICLR 2024): self-repair
    # gains are "often modest" once cost is counted.
    p_fix1: float = 0.5
    # [C] Debugging Decay Index (Adnan & Kuhn 2025, arXiv 2506.18403): most models
    # lose 60-80% of debugging effectiveness within 2-3 attempts -> fix 2 keeps 40%.
    fix_decay: float = 0.4
    max_fixes: int = 2                  # [D] cap from the DDI result
    restarts: int = 1                   # [D] one fresh restart, then hand over
    # [D] a script type is only routed locally if it passes >= 80% (success criterion)
    p_local_template: float = 0.8
    # [C bracket, P value] share of local answers re-asked to Haiku. RouteLLM
    # (Ong et al., ICLR 2025) kept 95% of GPT-4 quality with 14% strong-model calls on
    # MT Bench; Hybrid LLM (Ding et al., ICLR 2024) cut large-model calls by up to 40%
    # (i.e. 60% still large). 0.14-0.60, centre 0.4.
    p_question_fallback: float = 0.4

    # ---- local model energy, upper bound
    local_watts: float = 300            # [D] whole PC under load; RTX 5060 TGP alone is 145 W [C NVIDIA]
    local_seconds_question: float = 30  # [P] measure with Ollama eval timings
    local_seconds_script: float = 60    # [P]

    # ---- cost levers
    prompt_cache: bool = True           # [D] cache_control on every Claude call
    question_model: str = "haiku-5.5"
    script_model: str = "sonnet-5.5"

    # derived, current-tokenizer tokens
    @property
    def system(self): return self.system_prompt_old * self.uplift
    @property
    def context(self): return self.k * (self.chunk_old + self.chunk_label_old) * self.uplift
    @property
    def script(self): return self.script_old * self.uplift
    @property
    def answer(self): return self.answer_words * self.words_tokens_old * self.uplift
    @property
    def script_out(self): return self.script + self.script_notes_old * self.uplift + self.thinking_script


# ---------------------------------------------------------------- one call
@dataclass
class Call:
    step: str
    model: str
    fresh_in: float      # input tokens not in cache
    cached_in: float     # input tokens read from cache
    out: float
    prob: float = 1.0    # probability the call happens (expectation weight)
    cache_write: bool = True  # False for one-off calls: nothing would read the cache


    def usd(self, p: Params, batch: bool = False) -> float:
        pin, pout, pwrite, pread = PRICES[self.model]
        write_rate = pwrite if (p.prompt_cache and self.cache_write) else pin
        read_rate = pread if p.prompt_cache else pin
        cost = (self.fresh_in * write_rate + self.cached_in * read_rate + self.out * pout) / 1e6
        return cost * (BATCH_DISCOUNT if batch else 1.0)


def _chain(p: Params, first_in: float, first_out: float, first_step: str,
           start_prob: float, starts_with_fix: bool):
    """Sonnet loop: [write or first fix] -> fixes up to the cap -> restart(s).

    Each call in a chain re-sends the previous call's prompt plus its reply and
    the new error, so with caching on everything except the new tail is a cache
    read. Returns calls with the probability each one happens.
    """
    calls = []
    reach = start_prob
    prev_prompt = 0.0
    m = p.script_model
    # attempts: list of (step, pass_prob)
    plan = []
    if starts_with_fix:
        plan.append((first_step, p.p_fix1))
        fixes = [p.p_fix1 * p.fix_decay ** i for i in range(1, p.max_fixes)]
    else:
        plan.append((first_step, p.p_first))
        fixes = [p.p_fix1 * p.fix_decay ** i for i in range(p.max_fixes)]
    plan += [(f"fix {i + 1 + int(starts_with_fix)}", pf) for i, pf in enumerate(fixes)]
    for r in range(p.restarts):
        plan.append((f"restart {r + 1} write", p.p_first))
        plan += [(f"restart {r + 1} fix {i + 1}", p.p_fix1 * p.fix_decay ** i)
                 for i in range(p.max_fixes)]

    write_in = p.system + p.script_request + p.context
    for step, pass_prob in plan:
        if step == first_step:
            prompt = first_in
            fresh, cached = prompt, 0.0
            out = first_out
        elif "write" in step:              # fresh restart: history cleared
            prompt = write_in
            fresh, cached = p.script_request + p.context, p.system   # system prompt still cached
            out = p.script_out
        else:                              # fix: previous prompt + previous reply + error
            prompt = prev_prompt + p.script_out + p.error_text
            cached = prev_prompt
            fresh = prompt - cached
            out = p.script_out
        if not p.prompt_cache:
            fresh, cached = prompt, 0.0
        calls.append(Call(step, m, fresh, cached, out, reach))
        prev_prompt = prompt
        reach *= 1 - pass_prob
    return calls, reach   # reach = probability handed over to the user unsolved


def route_calls(p: Params, route: str):
    """Expected Claude calls for one request on a route. Returns (calls, local_seconds, p_unsolved)."""
    if route == "Q":
        q_in = p.system + p.question + p.context
        out = p.answer + p.thinking_question
        return [Call("Haiku fallback", p.question_model, q_in, 0.0, out, p.p_question_fallback, False)], \
            p.local_seconds_question, 0.0
    if route == "S1":
        return [], 0.0, 0.0
    if route == "S2":
        fail = 1 - p.p_local_template
        fix_in = p.system + p.script_request + p.context + p.script + p.error_text
        calls, unsolved = _chain(p, fix_in, p.script_out, "fix 1 (local script)", fail, True)
        return calls, p.local_seconds_script, unsolved
    if route == "S3":
        write_in = p.system + p.script_request + p.context
        calls, unsolved = _chain(p, write_in, p.script_out, "write", 1.0, False)
        return calls, 0.0, unsolved
    if route == "Q_claude":            # baseline: every question straight to Haiku
        q_in = p.system + p.question + p.context
        return [Call("answer", p.question_model, q_in, 0.0, p.answer + p.thinking_question, 1.0, False)], 0.0, 0.0
    raise ValueError(route)


def route_cost(p: Params, route: str, batch: bool = False):
    calls, secs, unsolved = route_calls(p, route)
    usd = sum(c.prob * c.usd(p, batch) for c in calls)
    kwh = p.local_watts * secs / 3.6e6
    pence = usd * GBP_PER_USD * 100 + kwh * PENCE_PER_KWH
    tin = sum(c.prob * (c.fresh_in + c.cached_in) for c in calls)
    tout = sum(c.prob * c.out for c in calls)
    ncalls = sum(c.prob for c in calls)
    return {"pence": pence, "claude_pence": usd * GBP_PER_USD * 100,
            "electricity_pence": kwh * PENCE_PER_KWH, "tokens_in": tin, "tokens_out": tout,
            "claude_calls": ncalls, "p_unsolved": unsolved}


# ------------------------------------------------------------ usage plan [D]
# Each row: (name, questions, scripts, script mix S1/S2/S3, hybrid?, batch?)
# Eval task set: 60 questions + 30 scripts, built as 10 per script route so every
# route is tested equally. 15 rounds = one per week of term 2 development.
EVAL_MIX = (1 / 3, 1 / 3, 1 / 3)
# Real use: no data yet. [P] Semantic-cache studies (Regmi & Pun, arXiv 2411.05276)
# saw 61.6-68.8% of queries match a cached one; we assume a lower 20% exact, 40% template.
USE_MIX = (0.2, 0.4, 0.4)
USAGE = [
    ("Pilot, Claude only (20 Q + 10 S)",           20, 10, (0, 0, 1), False, False),
    ("Hybrid eval, 15 rounds (60 Q + 30 S)",      900, 450, EVAL_MIX, True, True),
    ("Claude-only baseline, 3 rounds",            180,  90, (0, 0, 1), False, True),
    ("k sweep: k=3,5,8,10,20 x 60 Q x 3 rounds",  900,   0, EVAL_MIX, False, True),
    ("User study (10 people x 12 Q + 8 S)",       120,  80, USE_MIX, True, False),
    ("Ad hoc development",                        200, 100, USE_MIX, True, False),
]
K_SWEEP = (3, 5, 8, 10, 20)


def project_total(p: Params, batch=True, reuse=True, local_template=True, local_questions=True):
    """Project cost in GBP. The switches turn each lever off for the waterfall.

    Pilot and Claude-only baseline rows always run Claude only (they are the
    comparison), but batch and cache switches still apply to them.
    """
    rows, total = [], 0.0
    for name, nq, ns, mix, hybrid, batch_row in USAGE:
        b = batch and batch_row
        # Batch results can take longer than the 5-minute cache lifetime, so a fix
        # sent in the next batch would miss the cache. Conservative: no cache in batch rows.
        p_row = replace(p, prompt_cache=False) if b else p
        if name.startswith("k sweep"):
            # every k value on the same 60 questions, Haiku answering (no local step)
            per_k = nq / len(K_SWEEP)
            pence = sum(per_k * route_cost(replace(p_row, k=k), "Q_claude", b)["pence"]
                        for k in K_SWEEP)
        else:
            q_route = "Q" if (hybrid and local_questions) else "Q_claude"
            pence = nq * route_cost(p_row, q_route, b)["pence"]
            for share, r in zip(mix, ("S1", "S2", "S3")):
                use = r
                if not hybrid or (r == "S1" and not reuse) or (r == "S2" and not local_template):
                    use = "S3"
                pence += ns * share * route_cost(p_row, use, b)["pence"]
        rows.append((name, nq, ns, pence / 100))
        total += pence / 100
    return rows, total


def waterfall(p: Params):
    """Add one lever at a time, starting from Claude for everything."""
    off = dict(batch=False, reuse=False, local_template=False, local_questions=False)
    steps = [("Claude only, no levers", replace(p, prompt_cache=False), off)]
    steps.append(("+ Batch API on evaluation runs", replace(p, prompt_cache=False), {**off, "batch": True}))
    steps.append(("+ prompt caching", p, {**off, "batch": True}))
    steps.append(("+ return stored code for exact matches", p,
                  {**off, "batch": True, "reuse": True}))
    steps.append(("+ local LLM fills templates (Sonnet on fail)", p,
                  {**off, "batch": True, "reuse": True, "local_template": True}))
    steps.append(("+ local LLM answers questions (Haiku fallback)", p,
                  dict(batch=True, reuse=True, local_template=True, local_questions=True)))
    return [(label, project_total(pp, **kw)[1]) for label, pp, kw in steps]


SCENARIOS = {
    "optimistic": dict(p_first=0.7, p_fix1=0.6, fix_decay=0.4, p_local_template=0.9,
                       p_question_fallback=0.14, error_text=300),
    "central": {},
    "pessimistic": dict(p_first=0.3, p_fix1=0.3, fix_decay=0.2, p_local_template=0.5,
                        p_question_fallback=0.6, error_text=2000, thinking_script=2000),
}


def sensitivity(p: Params):
    """Swing in the central project total when one input moves to its optimistic
    and pessimistic value, the others staying central. Largest swing first."""
    base = project_total(p)[1]
    keys = sorted(set(SCENARIOS["optimistic"]) | set(SCENARIOS["pessimistic"]))
    out = []
    for key in keys:
        lo = SCENARIOS["optimistic"].get(key, getattr(p, key))
        hi = SCENARIOS["pessimistic"].get(key, getattr(p, key))
        t_lo = project_total(replace(p, **{key: lo}))[1]
        t_hi = project_total(replace(p, **{key: hi}))[1]
        out.append((key, lo, hi, t_lo, t_hi))
    out.sort(key=lambda r: -abs(r[4] - r[3]))
    return base, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = {}
    for name, kw in SCENARIOS.items():
        p = replace(Params(), **kw)
        per_route = {r: route_cost(p, r) for r in ("Q", "Q_claude", "S1", "S2", "S3")}
        rows, total = project_total(p)
        out[name] = {
            "per_route": per_route,
            "project_rows": rows,
            "project_total_gbp": total,
            "waterfall_gbp": waterfall(p),
        }
    base, sens = sensitivity(Params())
    out["sensitivity"] = {"central_gbp": base, "rows": sens}
    if args.json:
        print(json.dumps(out, indent=2))
        return
    p = Params()
    print("Token components (current tokenizer)")
    print(f"  system prompt {p.system:,.0f} | {p.k} pieces {p.context:,.0f} | question {p.question} | "
          f"script request {p.script_request} | answer {p.answer:,.0f} | script out {p.script_out:,.0f}")
    for name, o in out.items():
        if name == "sensitivity":
            continue
        print(f"\n== {name} ==")
        for r, c in o["per_route"].items():
            print(f"  {r:9s} {c['pence']:7.3f}p  (Claude {c['claude_pence']:.3f}p, "
                  f"power {c['electricity_pence']:.3f}p)  calls {c['claude_calls']:.2f}  "
                  f"in {c['tokens_in']:7,.0f} out {c['tokens_out']:6,.0f}  unsolved {c['p_unsolved']:.1%}")
        for row in o["project_rows"]:
            print(f"  {row[0]:45s} {row[1]:5d} Q {row[2]:4d} S  £{row[3]:6.2f}")
        print(f"  PROJECT TOTAL £{o['project_total_gbp']:.2f}")
        print("  Waterfall (same usage, one lever at a time)")
        for label, gbp in o["waterfall_gbp"]:
            print(f"    {label:48s} £{gbp:6.2f}")
    print("\nSensitivity of the central total (one input at a time)")
    for key, lo, hi, t_lo, t_hi in sens:
        print(f"  {key:22s} {lo!s:>6} -> £{t_lo:6.2f}   {hi!s:>6} -> £{t_hi:6.2f}")


if __name__ == "__main__":
    main()
