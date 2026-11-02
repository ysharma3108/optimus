"""Token and cost model for the OptimUS assistant (design 3: sort, look up, write, run, fix).

Every input is either MEASURED (by measure_corpus.py or the counts noted
below) or an ASSUMPTION that the pilot run will replace with logged
`response.usage` numbers. Change a value and re-run to see the effect.

    python tools/token_budget/cost_model.py
"""

# ---------------------------------------------------------------- tokenizer
# Counts below were made with the Claude tokenizer shipped in the anthropic
# SDK <= 0.34. Anthropic's token counting docs say models from Claude Opus 4.7
# on count "roughly 30 percent higher" than older models, and Haiku 5.5 and
# Sonnet 5.5 use the newer tokenizer, so all counts are scaled by this.
TOKENIZER_UPLIFT = 1.30

# ---------------------------------------------------------------- measured
SYSTEM_PROMPT = 596        # example_system_prompt.md (410 words)
ROUTER_PROMPT = 47         # one-line classification instruction
# ---------------------------------------------------------------- cited
# Mean user prompt length, Llama 2 tokenizer. Divided by the uplift because
# these are already counts of real text, not old-tokenizer counts.
QUESTION = 69.5 / 1.30        # LMSYS-Chat-1M, Zheng et al. 2024, Table 1 (arXiv 2309.11998)
SCRIPT_REQUEST = 295.6 / 1.30 # WildChat, Zhao et al. 2024, Table 1 (arXiv 2405.01470)
CHUNK = 384                # mean chunk, cell-aware chunking at a 512 target
CHUNK_LABEL = 15           # "[Notebook name #n]" tag around each chunk
SCRIPT_OUT = 849           # mean of the 3 starter scripts (655-1110)

# ---------------------------------------------------------------- assumptions
K_CHUNKS = 8               # retrieved chunks per call (to be tuned: 3-20)
ANSWER_OUT = 400           # 250-word cap x 1.44 tokens/word (measured on notebook prose) + citations
SCRIPT_NOTES = 100         # five lines of explanation after the script
ERROR_TEXT = 600           # traceback fed back on a failed run (range 200-2,000)
THINKING_QUESTION = 0      # Haiku 5.5 run with thinking off for questions
THINKING_SCRIPT = 1000     # Sonnet 5.5 low effort; high case 4,000
ROUTER_OUT = 5

PRICES = {  # USD per million tokens, platform.claude.com/docs/en/about-claude/pricing (Nov 2026)
    "haiku-5.5": (0.10, 0.50),
    "sonnet-5.5": (2.00, 10.00),
    "opus-5.5": (4.00, 20.00),
}
GBP_PER_USD = 0.75


def context(k=K_CHUNKS):
    return k * (CHUNK + CHUNK_LABEL)


def calls_question(k=K_CHUNKS):
    """Returns list of (step, tokens_in, tokens_out)."""
    return [
        ("sort", ROUTER_PROMPT + QUESTION, ROUTER_OUT),
        ("answer", SYSTEM_PROMPT + QUESTION + context(k), ANSWER_OUT + THINKING_QUESTION),
    ]


def calls_script(k=K_CHUNKS, fixes=0.75, thinking=THINKING_SCRIPT):
    write_in = SYSTEM_PROMPT + SCRIPT_REQUEST + context(k)
    write_out = SCRIPT_OUT + SCRIPT_NOTES + thinking
    fix_in = write_in + SCRIPT_OUT + ERROR_TEXT
    fix_out = write_out
    return [
        ("sort", ROUTER_PROMPT + SCRIPT_REQUEST, ROUTER_OUT),
        ("write", write_in, write_out),
        ("fixes", fixes * fix_in, fixes * fix_out),
    ]


def expected_fixes(p_first, p_fix, cap):
    """Expected number of fix calls when the first script passes with
    probability p_first, each fix passes with probability p_fix, and the
    loop stops after `cap` fixes."""
    e, fail = 0.0, 1 - p_first
    for _ in range(cap):
        e += fail
        fail *= 1 - p_fix
    return e


def total(calls):
    tin = sum(c[1] for c in calls) * TOKENIZER_UPLIFT
    tout = sum(c[2] for c in calls) * TOKENIZER_UPLIFT
    return tin, tout


def cost(tin, tout, model):
    pin, pout = PRICES[model]
    return (tin * pin + tout * pout) / 1e6


def main():
    up = TOKENIZER_UPLIFT
    print("Per-call breakdown (current-tokenizer estimate)")
    for name, calls in (("question", calls_question()), ("script", calls_script())):
        for step, i, o in calls:
            print(f"  {name:8s} {step:6s} in {i*up:8,.0f} out {o*up:7,.0f}")
        tin, tout = total(calls)
        print(f"  {name:8s} TOTAL  in {tin:8,.0f} out {tout:7,.0f}")

    print("\nExpected fixes per script")
    for label, p1, pf, cap in (("optimistic", 0.7, 0.6, 2), ("central", 0.5, 0.5, 2),
                               ("pessimistic", 0.3, 0.3, 3)):
        print(f"  {label:11s} p_first={p1} p_fix={pf} cap={cap} -> {expected_fixes(p1, pf, cap):.2f}")

    print("\nTokens and cost per request vs number of chunks k")
    for k in (3, 5, 8, 10, 20):
        qi, qo = total(calls_question(k))
        si, so = total(calls_script(k))
        print(f"  k={k:2d} question {qi:7,.0f}/{qo:5,.0f} "
              f"haiku {cost(qi, qo, 'haiku-5.5')*100*GBP_PER_USD:.3f}p | "
              f"script {si:7,.0f}/{so:5,.0f} sonnet {cost(si, so, 'sonnet-5.5')*100*GBP_PER_USD:.2f}p")

    # Usage built bottom-up
    usage = {
        "eval runs (60 Q + 30 S, 15 rounds)": (900, 450),
        "chunk sweep (5 k x 3 sizes x 60 Q)": (900, 0),
        "user study (10 people x 20 requests)": (120, 80),
        "ad hoc development": (200, 100),
    }
    nq = sum(v[0] for v in usage.values())
    ns = sum(v[1] for v in usage.values())
    print(f"\nUsage: {nq} questions, {ns} scripts")
    for k_, v in usage.items():
        print(f"  {k_}: {v}")

    print("\nProject cost (GBP), central case")
    qi, qo = total(calls_question())
    for fx_label, fx in (("optimistic", 0.42), ("central", 0.75), ("pessimistic", 1.53)):
        for th_label, th in (("think 1k", 1000), ("think 4k", 4000)):
            si, so = total(calls_script(fixes=fx, thinking=th))
            mix = (nq * cost(qi, qo, "haiku-5.5") + ns * cost(si, so, "sonnet-5.5")) * GBP_PER_USD
            hk = (nq * cost(qi, qo, "haiku-5.5") + ns * cost(si, so, "haiku-5.5")) * GBP_PER_USD
            sn = (nq * cost(qi, qo, "sonnet-5.5") + ns * cost(si, so, "sonnet-5.5")) * GBP_PER_USD
            print(f"  fixes {fx_label:11s} {th_label}: Haiku only £{hk:6.2f}  "
                  f"Haiku+Sonnet £{mix:6.2f}  Sonnet only £{sn:6.2f}")


if __name__ == "__main__":
    main()
