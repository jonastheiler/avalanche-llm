"""A rule-based baseline: keywords find the problem types, the training data fills in the rest.

python -m avalanche_llm.baseline

Any LLM has to beat this to be worth its cost.
"""

import json
import re
from collections import Counter

from avalanche_llm.config import PREDICTIONS_DIR
from avalanche_llm.dataset import load_split, save
from avalanche_llm.schema import Bulletin, Problem

# Checked against the training data: each pattern finds its type in most bulletins that have it.
KEYWORDS = {
    "new_snow": {"de": "neuschnee", "fr": "neige fraîche", "it": "neve fresca", "en": "new snow"},
    "wind_slab": {
        "de": "triebschnee",
        "fr": "neige soufflée|accumulations de neige",
        "it": "neve ventata|accumuli",
        "en": "wind slab|drifted snow|snowdrift",
    },
    "persistent_weak_layers": {
        "de": "altschnee|schwache schichten|schwachschicht",
        "fr": "neige ancienne|couches fragiles|vieille neige",
        "it": "neve vecchia|strati deboli",
        "en": "old snow|weak layer",
    },
    "wet_snow": {"de": "nass", "fr": "mouill", "it": "bagnat", "en": "wet"},
    "gliding_snow": {
        "de": "gleitschnee",
        "fr": "glissement",
        "it": "slittamento|scivolamento",
        "en": "glid",
    },
}
ABOVE = {
    "de": "oberhalb von rund",
    "fr": "au-dessus d'environ",
    "it": "al di sopra dei",
    "en": "above approximately",
}
BELOW = {
    "de": "unterhalb von rund",
    "fr": "en dessous d'environ",
    "it": "al di sotto dei",
    "en": "below approximately",
}
# Wet and gliding snow avalanches happen below an elevation, dry ones above it.
WET_TYPES = {"wet_snow", "gliding_snow"}


def most_common_problems(train: list[dict]) -> dict[str, Problem]:
    """The most frequent label of each problem type in the training data."""
    by_type: dict[str, Counter] = {}
    for example in train:
        for problem in example["label"]["problems"]:
            by_type.setdefault(problem["type"], Counter())[json.dumps(problem)] += 1
    return {t: Problem.model_validate_json(c.most_common(1)[0][0]) for t, c in by_type.items()}


def elevation(text: str, words: str) -> int | None:
    match = re.search(rf"{words}\s+(?:circa\s+)?(\d{{3,4}})\s?m", text)
    return int(match[1]) if match else None


def predict(text: str, lang: str, typical: dict[str, Problem]) -> Bulletin:
    types = [t for t, pattern in KEYWORDS.items() if re.search(pattern[lang], text, re.I)]
    lower, upper = elevation(text, ABOVE[lang]), elevation(text, BELOW[lang])
    problems = []
    for problem_type in types or ["no_distinct_avalanche_problem"]:
        problem = typical[problem_type].model_copy()
        if problem_type in WET_TYPES and upper:
            problem.elevation_upper = upper
        elif problem_type not in WET_TYPES and lower:
            problem.elevation_lower = lower
        problems.append(problem)
    return Bulletin(problems=problems)


def main() -> None:
    typical = most_common_problems(load_split("train"))
    predictions = [
        {
            "id": e["id"],
            "lang": e["lang"],
            "output": predict(e["text"], e["lang"], typical).model_dump_json(),
            "cost_usd": 0.0,
        }
        for e in load_split("test_sample")
    ]
    save(predictions, PREDICTIONS_DIR / "rules.jsonl")


if __name__ == "__main__":
    main()
