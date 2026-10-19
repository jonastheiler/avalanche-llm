from avalanche_llm.baseline import most_common_problems, predict
from avalanche_llm.schema import Problem

TYPICAL = {
    "wind_slab": Problem(type="wind_slab", danger_level=2, elevation_lower=2000),
    "gliding_snow": Problem(type="gliding_snow", danger_level=2),
    "no_distinct_avalanche_problem": Problem(type="no_distinct_avalanche_problem", danger_level=1),
}


def test_most_common_problems_picks_the_most_frequent_label_per_type():
    level = {"type": "wind_slab", "danger_level": 3}
    train = [{"label": {"problems": [level]}}] * 2 + [
        {"label": {"problems": [level | {"danger_level": 2}]}}
    ]

    assert most_common_problems(train)["wind_slab"].danger_level == 3


def test_predict_finds_types_and_puts_elevations_on_dry_and_wet_problems():
    text = (
        "Triebschnee oberhalb von rund 2400 m. "
        "Gleitschneelawinen an Grashängen unterhalb von rund 2600 m."
    )
    problems = predict(text, "de", TYPICAL).problems

    assert [p.type for p in problems] == ["wind_slab", "gliding_snow"]
    assert problems[0].elevation_lower == 2400
    assert problems[1].elevation_upper == 2600


def test_predict_falls_back_to_no_distinct_problem():
    problems = predict("Sunny and calm.", "en", TYPICAL).problems

    assert [p.type for p in problems] == ["no_distinct_avalanche_problem"]
    assert TYPICAL["wind_slab"].elevation_lower == 2000  # Typical problems are not changed.
