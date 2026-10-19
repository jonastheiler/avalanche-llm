"""Score predicted bulletins against the labels. Every approach is scored by this code.

Predicted and true problems are matched by their type, since a bulletin lists each type
at most once. The other fields are scored on the matched problems only, so they show how
well a model describes a problem it has found.
"""

from pydantic import ValidationError

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
    n_true = n_predicted = n_found = n_exact = 0
    field_hits = dict.fromkeys(FIELDS, 0)
    for label, prediction in zip(labels, predictions, strict=True):
        true = {p.type: p for p in label.problems}
        predicted = {p.type: p for p in prediction.problems} if prediction else {}
        n_true += len(true)
        n_predicted += len(predicted)
        for problem_type in true.keys() & predicted.keys():
            n_found += 1
            hits = field_matches(true[problem_type], predicted[problem_type])
            n_exact += all(hits.values())
            for field, hit in hits.items():
                field_hits[field] += hit

    return {
        "valid_json": sum(p is not None for p in predictions) / len(predictions),
        "problem_type_f1": f1(n_found, n_predicted, n_true),
        # A problem counts only if its type and every field are right: the headline score.
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
