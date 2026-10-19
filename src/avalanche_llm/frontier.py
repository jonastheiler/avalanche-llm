"""Extract the avalanche problems with Claude, through the Message Batches API.

python -m avalanche_llm.frontier claude-haiku-5-5                         # on validation
python -m avalanche_llm.frontier claude-opus-5-5 --shots 5 --split test_sample
python -m avalanche_llm.frontier claude-opus-5-5 --batch-id msgbatch_...   # resume

Needs ANTHROPIC_API_KEY. Batches cost half the normal price but can take up to an hour.
"""

import argparse
import json
import random
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from avalanche_llm.config import BATCH_DISCOUNT, PREDICTIONS_DIR, PRICES, SPLITS
from avalanche_llm.dataset import load_split, save
from avalanche_llm.schema import Bulletin

TASK = """\
You read Swiss avalanche bulletins by the WSL Institute for Snow and Avalanche Research SLF.
You get the avalanche problem descriptions of one bulletin. List every avalanche problem
the bulletin forecasts, each type at most once, with:
- type: the avalanche problem type
- danger_level: the danger level for this problem, 1 low, 2 moderate, 3 considerable,
  4 high, 5 very high
- danger_modifier: "+" or "-" if the level is in the upper or lower part of its range,
  else null"""

# The first prompt, scored on the test set before we knew better: it asks for empty aspects
# and null elevations when the text doesn't state them, but SLF's labels rarely have them.
NAIVE = f"""{TASK}
- aspects: the slope aspects where the problem occurs, empty if not specific
- elevation_lower / elevation_upper: the problem occurs above / below this elevation in
  metres, null if not bounded
Some fields are not stated in the text: give your best estimate from the description."""

# SLF's labelling conventions, measured on the training data.
CONVENTIONS = f"""{TASK}
- aspects: the slope aspects where the problem occurs
- elevation_lower / elevation_upper: the problem occurs above / below this elevation in
  metres, null if not bounded

Many fields are not stated in the text. Follow SLF's conventions:
- Dry problems (new_snow, wind_slab, persistent_weak_layers) almost always have an
  elevation_lower, most often 2000 to 2400 m, and no elevation_upper. Elevations in the
  text often describe snowfall, not the problem.
- Dry problems without aspects in the text usually cover all eight aspects.
  "Shady slopes" means N, NE, E, W, NW.
- wet_snow and gliding_snow never have an elevation_lower, usually no aspects, and almost
  never a danger_modifier. They sometimes have an elevation_upper.
- Wind slabs that form during a snowfall are part of the new_snow problem: list new_snow
  only, unless the text describes wind slabs as a separate problem.
- danger_modifier is used for about two thirds of the dry problems."""

PROMPTS = {"naive": NAIVE, "conventions": CONVENTIONS}

SCHEMA = anthropic.transform_schema(Bulletin)
# Extraction needs little reasoning. Low effort keeps thinking, and the bill, short.
EFFORT = "low"
MAX_TOKENS = 4096


def few_shot_examples(train: list[dict], lang: str, k: int, seed: int = 0) -> list[dict]:
    """k training examples in the given language that cover as many problem types as possible."""
    candidates = [e for e in train if e["lang"] == lang]
    random.Random(seed).shuffle(candidates)
    chosen, covered = [], set()
    for example in candidates:
        types = {p["type"] for p in example["label"]["problems"]}
        if len(chosen) < k and not types <= covered:
            chosen.append(example)
            covered |= types
    return chosen


def system_prompt(instructions: str, examples: list[dict]) -> str:
    shots = "".join(
        f"\n\n<example>\n<input>\n{e['text']}\n</input>\n"
        f"<output>{json.dumps(e['label'], ensure_ascii=False)}</output>\n</example>"
        for e in examples
    )
    return instructions + (f"\n\nExamples:{shots}" if shots else "")


def request(example: dict, model: str, system: str) -> Request:
    return Request(
        custom_id=f"{example['lang']}-{example['id']}",
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=MAX_TOKENS,
            # The system prompt is the same for every bulletin of a language: cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": example["text"]}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        ),
    )


def cost(usage, model: str) -> float:
    input_price, output_price = PRICES[model]
    input_tokens = (
        usage.input_tokens
        + 1.25 * (usage.cache_creation_input_tokens or 0)
        + 0.1 * (usage.cache_read_input_tokens or 0)
    )
    usd = (input_tokens * input_price + usage.output_tokens * output_price) / 1e6
    return usd * BATCH_DISCOUNT


def to_prediction(custom_id: str, result, model: str) -> dict:
    lang, bulletin_id = custom_id.split("-", 1)
    prediction = {"id": bulletin_id, "lang": lang, "output": None, "cost_usd": 0.0}
    if result.type != "succeeded":
        return prediction
    message = result.message
    prediction["cost_usd"] = cost(message.usage, model)
    # A refusal or a cut-off answer may not match the schema: count it as invalid.
    if message.stop_reason == "end_turn":
        prediction["output"] = next((b.text for b in message.content if b.type == "text"), None)
    return prediction


def run_batch(client: anthropic.Anthropic, requests: list[Request]) -> str:
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch {batch.id} created: resume with --batch-id {batch.id}")
    return batch.id


def wait(client: anthropic.Anthropic, batch_id: str) -> None:
    while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        print(f"{batch.request_counts.processing} requests still processing")
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=list(PRICES))
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--prompt", choices=list(PROMPTS), default="conventions")
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--batch-id", help="collect an earlier batch instead of starting one")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    if not (batch_id := args.batch_id):
        train, test = load_split("train"), load_split(args.split)
        systems = {
            lang: system_prompt(PROMPTS[args.prompt], few_shot_examples(train, lang, args.shots))
            for lang in {e["lang"] for e in test}
        }
        batch_id = run_batch(client, [request(e, args.model, systems[e["lang"]]) for e in test])
    wait(client, batch_id)

    predictions = [
        to_prediction(r.custom_id, r.result, args.model)
        for r in client.messages.batches.results(batch_id)
    ]
    name = f"{args.model}-{args.shots}shot-{args.prompt}.jsonl"
    save(predictions, PREDICTIONS_DIR / args.split / name)
    total = sum(p["cost_usd"] for p in predictions)
    print(f"{len(predictions)} predictions, {total:.2f} US$")


if __name__ == "__main__":
    main()
