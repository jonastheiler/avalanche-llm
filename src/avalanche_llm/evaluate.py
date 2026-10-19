"""Score predicted bulletins against the labels. Every approach is scored by this code.

python -m avalanche_llm.evaluate                      # every file in data/predictions/test_sample/
python -m avalanche_llm.evaluate --split validation

Predicted and true problems are matched by their type, since a bulletin lists each type
at most once. The other fields are scored on the matched problems only, so they show how
well a model describes a problem it has found.
"""

import argparse

from pydantic import ValidationError

from avalanche_llm.config import LANGUAGES, PREDICTIONS_DIR, SPLITS
from avalanche_llm.dataset import load_split, read
from avalanche_llm.schema import Bulletin, Problem

FIELDS = ["danger_level", "danger_modifier", "aspects", "elevation_lower", "elevation_upper"]


def parse(output: str) -> Bulletin | None:
    """A model's raw output as a Bulletin, or None if it isn't valid JSON for the schema."""
    try:
        return Bulletin.model_validate_json(output)
    except ValidationError:
        return None


def f1(true_positives: int, n_predicted: int, n_true: int) -> float:
    return 2 * true_positives / (n_predicted + n_true) if n_predicted + n_true else 1.0


def score(labels: list[Bulletin], predictions: list[Bulletin | None]) -> dict[str, float]:
    """Invalid predictions count as bulletins without problems: every true problem is missed."""
    n_true = n_predicted = n_found = n_with_level = n_exact = 0
    field_hits = dict.fromkeys(FIELDS, 0)
    for label, prediction in zip(labels, predictions, strict=True):
        true = {p.type: p for p in label.problems}
        predicted = {p.type: p for p in prediction.problems} if prediction else {}
        n_true += len(true)
        n_predicted += len(predicted)
        for problem_type in true.keys() & predicted.keys():
            n_found += 1
            hits = field_matches(true[problem_type], predicted[problem_type])
            n_with_level += hits["danger_level"]
            n_exact += all(hits.values())
            for field, hit in hits.items():
                field_hits[field] += hit

    return {
        "valid_json": sum(p is not None for p in predictions) / len(predictions),
        # The headline: a problem counts if its type and danger level are right. Several
        # other fields are often not in the text at all, so requiring them too would leave
        # every approach near zero.
        "type_level_f1": f1(n_with_level, n_predicted, n_true),
        "problem_type_f1": f1(n_found, n_predicted, n_true),
        # Stricter: a problem counts only if its type and every field are right.
        "exact_problem_f1": f1(n_exact, n_predicted, n_true),
        **{field: hits / n_found if n_found else 0.0 for field, hits in field_hits.items()},
    }


def field_matches(true: Problem, predicted: Problem) -> dict[str, bool]:
    return {
        "danger_level": predicted.danger_level == true.danger_level,
        "danger_modifier": predicted.danger_modifier == true.danger_modifier,
        "aspects": set(predicted.aspects) == set(true.aspects),
        "elevation_lower": predicted.elevation_lower == true.elevation_lower,
        "elevation_upper": predicted.elevation_upper == true.elevation_upper,
    }


def results(test: list[dict], predictions: list[dict]) -> dict[str, float]:
    """Scores on the whole test sample, the headline score per language and the cost."""
    by_key = {(p["id"], p["lang"]): p for p in predictions}
    found = [by_key[e["id"], e["lang"]] for e in test]
    labels = [Bulletin.model_validate(e["label"]) for e in test]
    parsed = [parse(p["output"]) if p["output"] else None for p in found]
    per_language = {
        f"type_level_f1_{lang}": score(
            [label for label, e in zip(labels, test, strict=True) if e["lang"] == lang],
            [pred for pred, e in zip(parsed, test, strict=True) if e["lang"] == lang],
        )["type_level_f1"]
        for lang in LANGUAGES
    }
    cost = 1000 * sum(p["cost_usd"] for p in found) / len(found)
    return score(labels, parsed) | per_language | {"usd_per_1000_bulletins": cost}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, default="test_sample")
    split = parser.parse_args().split
    test = load_split(split)
    paths = sorted((PREDICTIONS_DIR / split).glob("*.jsonl"))
    table = {path.stem: results(test, read(path)) for path in paths}
    metrics = list(next(iter(table.values())))
    print("| metric | " + " | ".join(table) + " |")
    print("|---" * (len(table) + 1) + "|")
    for metric in metrics:
        print(f"| {metric} | " + " | ".join(f"{r[metric]:.3f}" for r in table.values()) + " |")


if __name__ == "__main__":
    main()
