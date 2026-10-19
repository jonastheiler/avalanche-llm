# avalanche-llm

**How many labelled examples does a small open model need before it matches a frontier
LLM at reading Swiss avalanche bulletins, and what does each option cost?**

Work in progress. The plan:

1. **Data:** every bulletin since November 2023 from the SLF API, in German, French,
   Italian and English. The bulletins' own structured fields are the labels.
2. **Task:** from an avalanche-problem description, predict the danger level, the
   problem type, the aspects and the elevation range as JSON.
3. **Compared:** a rule-based baseline, a frontier LLM (zero-shot and few-shot) and a
   small open model fine-tuned with LoRA on 100, 300 and 1,000 examples.
4. **Measured:** per-field accuracy, valid JSON, cost per 1,000 bulletins, latency and
   results per language.

## Getting started

```
uv sync
uv run python -m avalanche_llm.download   # about 2 minutes
uv run python -m avalanche_llm.dataset    # train: winters 23/24 + 24/25, test: 25/26
uv run pytest
```

Run the approaches on the same test sample (300 bulletins per language) and compare them:

```
uv run python -m avalanche_llm.baseline                          # rules, free
uv run python -m avalanche_llm.frontier claude-haiku-5-5 --shots 0   # needs ANTHROPIC_API_KEY
uv run python -m avalanche_llm.frontier claude-opus-5-5 --shots 5
uv run python -m avalanche_llm.evaluate                          # results table
```

Predictions are committed in `data/predictions/`, so the results table can be
reproduced without paying for the LLM calls again.

## Data

Avalanche bulletins: [WSL Institute for Snow and Avalanche Research SLF](https://www.slf.ch),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
