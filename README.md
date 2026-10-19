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

Prompts are developed on a validation sample (100 bulletins per language, held out from
the 24/25 winter). Only the final runs use the test sample (300 bulletins per language):

```
uv run python -m avalanche_llm.baseline                       # rules, free, both samples
uv run python -m avalanche_llm.frontier claude-haiku-5-5      # needs ANTHROPIC_API_KEY
uv run python -m avalanche_llm.evaluate --split validation    # compare prompts
uv run python -m avalanche_llm.frontier claude-opus-5-5 --shots 5 --split test_sample
uv run python -m avalanche_llm.evaluate                       # results on the test sample
```

Fine-tune Qwen3-1.7B with LoRA on a GPU, e.g. a free Kaggle notebook with a T4
(the newest PyTorch versions no longer support Kaggle's P100):

```
pip install -e ".[finetune]"
python -m avalanche_llm.dataset
python -m avalanche_llm.finetune 16 --smoke        # a few minutes: checks the setup
python -m avalanche_llm.finetune 0 100 300 1000    # 0 = without fine-tuning
```

The headline score is **type + level F1**: a predicted avalanche problem counts if both its
type and its danger level are right.

Predictions are committed in `data/predictions/`, so the results table can be
reproduced without paying for the LLM calls again.

## Data

Avalanche bulletins: [WSL Institute for Snow and Avalanche Research SLF](https://www.slf.ch),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
