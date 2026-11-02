# Token budget for the OptimUS assistant

Reproducible numbers behind the assistant's token and cost estimate.

| File | What it is |
| --- | --- |
| `measure_corpus.py` | Chunks the 24 notebooks (markdown + code cells, outputs dropped) and counts tokens with a real tokenizer. |
| `measured_corpus.json` | Output of the run used in the design proposal. |
| `example_system_prompt.md` | Draft instructions for the assistant, counted as the "system prompt". |
| `example_requests.txt` | 10 example questions (Q) and 5 example script requests (S). |
| `cost_model.py` | The per-request token model and project cost, with every input labelled measured or assumed. |

```
pip install tokenizers
python tools/token_budget/measure_corpus.py --tokenizer anthropic/tokenizer.json --extra path/to/scripts/*.py
python tools/token_budget/cost_model.py
```

The tokenizer file is `anthropic/tokenizer.json` from the `anthropic==0.34.0` wheel
(`pip download anthropic==0.34.0 --no-deps`, then unzip). Newer Claude models count about
30% more tokens for the same text, so `cost_model.py` scales every count by 1.3. Once an
API key is available, replace both with the free `messages.count_tokens` endpoint and the
`usage` block that every response returns.
