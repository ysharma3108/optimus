# OptimUS assistant

A documentation-grounded assistant for OptimUS: cited answers to questions,
and starter scripts that have actually run. The design and every number in
it are explained in the architecture plan (Claude Doc "OptimUS AI Assistant:
Architecture Plan"); all settings live in [`config.py`](config.py), each
labelled CITED, OPTIMUS, DESIGN, PILOT or MEASURED.

## Four routes

| Route | When | Model |
| --- | --- | --- |
| 0 repeat | the same request was answered before | none |
| 1 question | a how/what/why question | local model, Haiku if no citation or weak match |
| 2 template | a script close to a verified example | local model fills values, Sonnet fixes |
| 3 new script | no close match | Sonnet: write, check, run, at most 2 fixes, 1 fresh restart |

## Modules

| Module | Job |
| --- | --- |
| `ingest.py` | notebooks, source and docs into cited pieces; list of real API names |
| `retrieve.py` | BM25 search, optional dense ranking merged by Reciprocal Rank Fusion |
| `router.py` | picks the route |
| `answer.py` | cited answers, fallback when citations are missing |
| `checks.py` | syntax, made-up API names, elements per wavelength |
| `loop.py` | write, check, run, fix loop with the cited caps |
| `runner.py` | runs a script in the `optimuslib/optimus` Docker image |
| `cache.py` | exact repeat cache (0 tokens) |
| `tokenlog.py` | real token counts per call from the API `usage` field |
| `llm.py` | one interface for Claude, Ollama and a test fake |
| `metrics.py` | Wilson intervals, hit rate sweep, best k, McNemar test |

## Use (PowerShell or bash, from the repository root)

```
python -m assistant index
python -m assistant search "how do I create a bowl source"
python -m assistant route "Write a script for a 500 kHz bowl transducer in water"
python -m assistant check path\to\script.py
python -m pytest assistant
```

Needs Python 3.9 or newer on the host and nothing else; the solver itself
only runs inside Docker. `ask` additionally needs Ollama
(`ollama pull qwen3.5:9b`) or `pip install anthropic` and an API key.

Run tests with `python -m pytest assistant`, not plain `pytest` from the
root: the root `setup.cfg` makes pytest import OptimUS, which needs bempp.

## Baseline so far

BM25 alone on the 12 seed questions in `eval/seed_questions.jsonl`
(measured 9 Nov 2026): a correct source in the top 8 for 8 of 12, top 20 for
12 of 12. This is a starting point, not a target; the pilot sets the targets.
