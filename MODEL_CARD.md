---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
language: [de, fr, it, en]
tags: [lora, information-extraction, structured-output, avalanche, switzerland]
---

# Qwen3-1.7B avalanche LoRA

A LoRA adapter for [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) that reads the
avalanche problem descriptions of a Swiss avalanche bulletin (SLF) and returns the
avalanche problems as JSON: type, danger level, aspects and elevation range. It works in
German, French, Italian and English.

> **Research only. Do not use it for decisions in avalanche terrain.** About 40% of the
> problems it predicts are wrong on type or danger level. For the actual danger, read the
> official bulletin at [slf.ch](https://www.slf.ch).

It was trained on 1,000 bulletins to answer one question: how many labelled examples does
a small open model need to match a frontier LLM at structured extraction? Code, data
splits and all predictions are in
[github.com/jonastheiler/avalanche-llm](https://github.com/jonastheiler/avalanche-llm).

## Results

Test sample: 1,200 bulletins from winter 25/26 (300 per language), later than all
training data. A predicted problem counts as correct if both its type and danger level
match the forecasters' label.

| | type + level F1 (95% CI) | danger level | valid JSON | USD per 1,000 bulletins |
|---|---|---|---|---|
| keyword rules | 0.253 (0.224–0.284) | 0.339 | 1.000 | 0 |
| Qwen3-1.7B, no fine-tuning | 0.000 | 0.000 | 0.000 | 0.23 |
| **this adapter (1,000 examples)** | **0.604 (0.573–0.638)** | **0.785** | **1.000** | **0.12** |
| Claude Haiku 5.5, zero-shot | 0.597 (0.569–0.629) | 0.750 | 1.000 | 0.17 |
| Claude Haiku 5.5, 5-shot | 0.610 (0.580–0.640) | 0.750 | 1.000 | 0.13 |

The adapter ties Claude Haiku: the intervals overlap almost completely. Results are about
the same in all four languages. The cost assumes a rented T4 at US$0.40 per hour.

## Usage

The adapter expects the same system prompt it was trained with, which lives in the
GitHub repo:

```python
# pip install torch transformers peft git+https://github.com/jonastheiler/avalanche-llm
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from avalanche_llm.frontier import CALIBRATED

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "jonastheiler/qwen3-1.7b-avalanche-lora")

text = "As a consequence of bise wind, mostly small wind slabs formed on Thursday. ..."
messages = [{"role": "system", "content": CALIBRATED}, {"role": "user", "content": text}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The output looks like this:

```json
{"problems": [{"type": "wind_slab", "danger_level": 2, "danger_modifier": null,
  "aspects": ["N", "NE", "E", "NW"], "elevation_lower": 2200, "elevation_upper": null}]}
```

Problem types: `new_snow`, `wind_slab`, `persistent_weak_layers`, `wet_snow`,
`gliding_snow`, `no_distinct_avalanche_problem`.

## Training

- **Data:** 1,000 bulletins from winters 23/24 and 24/25, mixed languages. The labels are
  the structured fields the SLF forecasters publish with each bulletin, so no hand
  labelling was needed.
- **LoRA:** r 16, alpha 32, dropout 0.05, on all attention and MLP projections.
- **Training:** 3 epochs, AdamW, learning rate 2e-4, batch size 8, fp16 on one T4
  (about 40 minutes). The loss covers only the answer, not the prompt. The loop is plain
  PyTorch, in `finetune.py` in the repo.

## Limitations

- **Not accurate enough to rely on.** Fewer than 1 in 10 predicted problems are right in every field.
- **The text often doesn't contain the label.** Forecasters set aspects and elevations
  directly, and the text frequently doesn't state them, so the model's values for these
  are mostly educated guesses that match SLF's typical conventions.
- **Only tested on SLF bulletins.** Other avalanche services, older archives or field
  reports may be worded differently.
- **One training run with one seed.** The test intervals are about ±0.03.

## Data

Avalanche bulletins: [WSL Institute for Snow and Avalanche Research SLF](https://www.slf.ch),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
