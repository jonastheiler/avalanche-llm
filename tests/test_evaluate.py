import pytest

from avalanche_llm.evaluate import confidence_interval, parse, score
from avalanche_llm.schema import Bulletin, Problem

WIND_SLAB = Problem(type="wind_slab", danger_level=3, aspects=["N", "NE"], elevation_lower=2400)
GLIDING = Problem(type="gliding_snow", danger_level=2)


def test_parse_rejects_invalid_output():
    assert parse('{"problems": []}') == Bulletin(problems=[])
    assert parse("not json") is None
    assert parse('{"problems": [{"type": "avalanche"}]}') is None


def test_perfect_predictions_score_one():
    labels = [Bulletin(problems=[WIND_SLAB, GLIDING])]

    assert set(score(labels, labels).values()) == {1.0}


def test_fields_are_scored_on_matched_problems_only():
    labels = [Bulletin(problems=[WIND_SLAB, GLIDING])]
    # Wind slab found with the wrong level; gliding snow missed.
    predictions = [Bulletin(problems=[WIND_SLAB.model_copy(update={"danger_level": 2})])]

    result = score(labels, predictions)

    assert result["problem_type_f1"] == pytest.approx(2 / 3)
    assert result["type_level_f1"] == 0.0
    assert result["exact_problem_f1"] == 0.0
    assert result["danger_level"] == 0.0
    assert result["aspects"] == 1.0


def test_type_level_f1_ignores_the_other_fields():
    labels = [Bulletin(problems=[WIND_SLAB])]
    predictions = [Bulletin(problems=[WIND_SLAB.model_copy(update={"aspects": []})])]

    result = score(labels, predictions)

    assert result["type_level_f1"] == 1.0
    assert result["exact_problem_f1"] == 0.0


def test_invalid_prediction_misses_every_problem():
    labels = [Bulletin(problems=[WIND_SLAB]), Bulletin(problems=[GLIDING])]

    result = score(labels, [None, Bulletin(problems=[GLIDING])])

    assert result["valid_json"] == 0.5
    assert result["problem_type_f1"] == pytest.approx(2 / 3)


def test_confidence_interval_contains_the_score_and_is_narrow_when_all_agree():
    labels = [Bulletin(problems=[WIND_SLAB])] * 4 + [Bulletin(problems=[GLIDING])] * 4
    perfect = confidence_interval(labels, labels, list("aabbccdd"))
    half_right = confidence_interval(labels, [labels[0]] * 8, list("aabbccdd"))

    assert perfect == (1.0, 1.0)
    assert half_right[0] < 0.5 < half_right[1]
