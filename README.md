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
uv run pytest
```

## Data

Avalanche bulletins: [WSL Institute for Snow and Avalanche Research SLF](https://www.slf.ch),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
