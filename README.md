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

## Results so far

On the test sample (1,200 bulletins from winter 25/26, 300 per language), Qwen3-1.7B
fine-tuned on 1,000 examples matches Claude Haiku:

| | type + level F1 (95% CI) | danger level | valid JSON | USD per 1,000 bulletins |
|---|---|---|---|---|
| rules | 0.253 (0.224–0.284) | 0.339 | 1.000 | 0 |
| Qwen3-1.7B, no fine-tuning | 0.000 | 0.000 | 0.000 | 0.23 |
| Qwen3-1.7B + LoRA, 100 examples | 0.367 (0.334–0.401) | 0.597 | 1.000 | 0.12 |
| Qwen3-1.7B + LoRA, 300 examples | 0.516 (0.483–0.549) | 0.743 | 0.998 | 0.14 |
| Qwen3-1.7B + LoRA, 1,000 examples | 0.604 (0.573–0.638) | 0.785 | 1.000 | 0.12 |
| Claude Haiku 5.5, zero-shot | 0.597 (0.569–0.629) | 0.750 | 1.000 | 0.17 |
| Claude Haiku 5.5, 5-shot | 0.610 (0.580–0.640) | 0.750 | 1.000 | 0.13 |

How to read the columns:

- **type + level F1:** a bulletin describes one or more avalanche problems (e.g. wind
  slab, wet snow). A predicted problem counts as correct if both its type and its danger
  level (1–5) match the bulletin. F1 balances problems that were missed against problems
  that were made up. The 95% interval comes from resampling the bulletins, with each
  bulletin and its translations resampled together.
- **danger level:** of the problems whose type was found, the share with the right
  danger level.
- **valid JSON:** the share of answers that parse into the expected schema. An invalid
  answer counts as a bulletin with no problems.
- **USD per 1,000 bulletins:** for Claude, the API price. For Qwen, inference time priced
  at US$0.40 per hour for a rented T4, not counting training.

### An example

A bulletin from 27 November 2025 (English version), as the models see it:

> A lot of snow has fallen since Sunday. The new snow is in some cases still prone to
> triggering.
> As a consequence of bise wind, mostly small wind slabs formed on Thursday. These are to
> be evaluated with care and prudence in particular in very steep terrain.
> Careful route selection is recommended.
>
> Gliding avalanches are possible on steep grassy slopes. These can in isolated cases
> reach medium size.

The label, set by the SLF forecasters:

```json
{"problems": [
  {"type": "new_snow", "danger_level": 2, "danger_modifier": null,
   "aspects": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
   "elevation_lower": 1600, "elevation_upper": null},
  {"type": "gliding_snow", "danger_level": 1, "danger_modifier": null,
   "aspects": [], "elevation_lower": null, "elevation_upper": null}
]}
```

The predicted problems, as type and danger level:

| | new snow | wind slab | gliding snow |
|---|---|---|---|
| **label** | **2** | – | **1** |
| rules | 3 | 1 | 1 ✓ |
| Qwen, no fine-tuning | – (no JSON) | – | – |
| Qwen + LoRA, 100 | – | 2 | 1 ✓ |
| Qwen + LoRA, 1,000 | 3 | – | 1 ✓ |
| Haiku, 5-shot | 3 | 2 | 1 ✓ |

Every approach gets the gliding snow right. Haiku and the rules also report wind slabs,
because the text describes them, but the forecasters didn't list them as a problem.
Qwen with 1,000 examples has learnt to leave them out, while Qwen with 100 examples
mistakes the new snow for wind slabs. Nobody gets the new-snow level: the text gives
no clear hint that it is 2 rather than 3. The elevation of 1,600 m isn't in the text
at all. Qwen guessed 2,200 m and Haiku 2,000 m.

### What the results show

- **Fine-tuning is what makes the small model usable.** Without it, Qwen answers in a list
  format rather than JSON and scores zero. With 100 examples it follows the format but is
  well behind Haiku. With 300 it is close, and with 1,000 it is level.
- **Qwen with 1,000 examples ties Haiku.** Its F1 of 0.604 lies between Haiku zero-shot
  (0.597) and 5-shot (0.610), and the three intervals overlap almost completely. Qwen is
  slightly better at the danger level (0.785 vs 0.750).
- **The cost is about the same.** Both cost around US$0.12–0.17 per 1,000 bulletins, so
  the case for the small model is control (it runs anywhere and the data stays local),
  not price.
- **The danger level needs a model.** Keyword rules find the problem type almost as
  well as Haiku (F1 0.746 vs 0.796), but they fill in the danger level with the most
  frequent level for that type in the training data, which is right only a third of
  the time.
- **No language is left behind.** Results are about the same in German, French, Italian
  and English.

### Are the results useful?

As a comparison of approaches, yes. As an avalanche tool, not on their own.

What holds up:

- **The comparison is fair.** All approaches are scored by the same code on the same
  bulletins. The test winter (25/26) is later than every training example, so a model
  can't have memorised its bulletins, and the prompts were tuned on a separate
  validation sample.
- **The main finding is clear.** The learning curve (0 → 100 → 300 → 1,000 examples)
  climbs steeply and reaches Haiku's level, and the gaps between the steps are much
  larger than the intervals. A 1.7B model that runs on a free GPU can take over a
  structured extraction task from an API model with about 1,000 labelled examples.
- **The labels cost nothing here.** The SLF publishes the structured fields alongside the
  text, so the 1,000 examples were free. For most tasks, labelling them by hand would
  be the main cost of the small-model route, and this sets a rough budget for it.

What limits them:

- **No approach is accurate enough to rely on.** Even the best get about 40% of the
  avalanche problems wrong on type or level, and fewer than 1 in 10 are right in every
  field.
- **Part of the gap can't be closed from the text.** The forecasters set the structured
  fields directly, and the text often doesn't contain them. In the first test bulletin
  the text says "above about 3,500 m" while the label says 3,400 m, and it names no
  aspects at all. So the low aspect and elevation scores say more about the task than
  about the models, and nobody knows the best score that is achievable.
- **The SLF doesn't need this.** It already publishes the structured fields. The
  approach would matter for text without structured data, such as older archives,
  other avalanche services or field reports, but those weren't tested and may be
  worded differently.
- **It is one run on one sample.** Qwen was fine-tuned once per size with one random
  seed, and 300 bulletins (each in four languages) give intervals of about ±0.03. Qwen
  and Haiku are tied, but neither can be called better.
- **The frontier comparison is incomplete.** Haiku is Claude's smallest model. Opus may
  well score higher, so "matches a frontier LLM" holds only for Haiku so far.
- **The costs are estimates.** Qwen's assumes a rented T4 at US$0.40 per hour and leaves
  out training (about 40 minutes for 1,000 examples, well under US$1). At these prices
  cost doesn't decide between the two either way.

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
pip uninstall -y torchao       # Kaggle only: its old torchao makes peft refuse to load
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
