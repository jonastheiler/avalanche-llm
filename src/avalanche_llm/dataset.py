"""Turn the raw bulletins into deduplicated train and test examples.

python -m avalanche_llm.dataset

One example is one bulletin: its avalanche problem descriptions as input, its avalanche
problems as label. Problems are not used as examples on their own, because several
problems often share one description, and their labels would then contradict each other.
"""

import html
import json
import re
from pathlib import Path

from avalanche_llm.config import LANGUAGES, PROCESSED_DIR, TEST_WINTER
from avalanche_llm.download import load
from avalanche_llm.schema import Bulletin, Problem

DANGER_LEVELS = {"low": 1, "moderate": 2, "considerable": 3, "high": 4, "very_high": 5}
# "neutral" (e.g. 2=) and a missing subdivision both mean "no + or -": the descriptions
# rarely tell them apart, so we merge them.
MODIFIERS = {"plus": "+", "minus": "-"}
ASPECT_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def clean(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", html.unescape(text))
    return re.sub(r" ?\n ?", "\n", text).strip()


def winter(timestamp: str) -> str:
    """'2024-02-10T07:00:00Z' -> '2023/24'. A winter runs from September to August."""
    year, month = int(timestamp[:4]), int(timestamp[5:7])
    start = year if month >= 9 else year - 1
    return f"{start}/{(start + 1) % 100:02d}"


def to_problem(raw: dict) -> Problem:
    subdivision = (raw.get("customData") or {}).get("CH", {}).get("subdivision")
    elevation = raw.get("elevation") or {}
    lower, upper = elevation.get("lowerBound"), elevation.get("upperBound")
    return Problem(
        type=raw["problemType"],
        danger_level=DANGER_LEVELS[raw["dangerRatingValue"]],
        danger_modifier=MODIFIERS.get(subdivision),
        aspects=[a for a in ASPECT_ORDER if a in (raw.get("aspects") or [])],
        elevation_lower=int(lower) if lower is not None else None,
        elevation_upper=int(upper) if upper is not None else None,
    )


def to_example(bulletin: dict) -> dict | None:
    raw_problems = bulletin["avalancheProblems"]
    if len({p["problemType"] for p in raw_problems}) < len(raw_problems):
        return None  # A problem type listed twice: once in the morning, once later.
    # Only the problem descriptions. The bulletin also holds "coreZoneText", which spells
    # out the labels ("Danger level 3+ on all aspects above 2000m"): using it would leak them.
    descriptions = dict.fromkeys(clean(p["comment"]) for p in raw_problems)
    label = Bulletin(problems=[to_problem(p) for p in raw_problems])
    return {
        "id": bulletin["bulletinID"],
        "lang": bulletin["lang"],
        "winter": winter(bulletin["validTime"]["startTime"]),
        "valid_from": bulletin["validTime"]["startTime"],
        "regions": [r["regionID"] for r in bulletin["regions"]],
        "text": "\n\n".join(descriptions),
        "label": label.model_dump(),
    }


def deduplicate(examples: list[dict]) -> list[dict]:
    """Keep the first example of every text. SLF reuses texts across regions and days.
    Drop texts that come with different labels: no model could get them right."""
    by_text: dict[str, list[dict]] = {}
    for example in sorted(examples, key=lambda e: e["valid_from"]):
        by_text.setdefault(example["text"], []).append(example)
    return [
        group[0]
        for group in by_text.values()
        if all(e["label"] == group[0]["label"] for e in group)
    ]


def split(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [e for e in examples if e["winter"] != TEST_WINTER]
    seen = {e["text"] for e in train}
    test = [e for e in examples if e["winter"] == TEST_WINTER and e["text"] not in seen]
    return train, test


def save(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_split(name: str) -> list[dict]:
    with open(PROCESSED_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    train, test = [], []
    for lang in LANGUAGES:
        examples = [e for i in load(lang) for b in i["bulletins"] if (e := to_example(b))]
        lang_train, lang_test = split(deduplicate(examples))
        train += lang_train
        test += lang_test
        print(f"{lang}: {len(lang_train)} train, {len(lang_test)} test examples")
    save(train, PROCESSED_DIR / "train.jsonl")
    save(test, PROCESSED_DIR / "test.jsonl")


if __name__ == "__main__":
    main()
