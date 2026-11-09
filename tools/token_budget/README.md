# Token budget for the OptimUS assistant

Reproducible numbers behind the assistant's token and cost estimate.

| File | What it is |
| --- | --- |
| `measure_corpus.py` | Chunks the 24 notebooks (markdown + code cells, outputs dropped) and counts tokens with a real tokenizer. |
| `measured_corpus.json` | Output of the run used in the design proposal. |
| `example_system_prompt.md` | Draft instructions for the assistant, counted as the "system prompt". |
| `example_requests.txt` | 10 example questions (Q) and 5 example script requests (S). |
| `cost_model.py` | The per-request token model and project cost, with every input labelled measured or assumed (Claude-only design). |
| `hybrid_cost_model.py` | Cost model for the hybrid local/Claude design (4 routes, fix loop with restart, prompt cache, Batch API, local electricity). Every input tagged [M] measured, [C] cited, [D] by design or [P] pilot. Prints per-route cost, project total, a lever-by-lever waterfall and a sensitivity table. |
| `test_hybrid_cost_model.py` | Unit tests: hand-calculated prices, loop call counts, worst case, scenario ordering. |
| `hybrid_cost_model_output.txt` | Output of the run used in the cost model doc (9 Nov 2026). |

```
pip install tokenizers
python tools/token_budget/measure_corpus.py --tokenizer anthropic/tokenizer.json --extra path/to/scripts/*.py
python tools/token_budget/cost_model.py
python tools/token_budget/hybrid_cost_model.py
python -m unittest tools/token_budget/test_hybrid_cost_model.py
```

The tokenizer file is `anthropic/tokenizer.json` from the `anthropic==0.34.0` wheel
(`pip download anthropic==0.34.0 --no-deps`, then unzip). Newer Claude models count about
30% more tokens for the same text, so `cost_model.py` scales every count by 1.3. Once an
API key is available, replace both with the free `messages.count_tokens` endpoint and the
`usage` block that every response returns.
