from avalanche_llm.dataset import clean, deduplicate, split, to_example, winter


def make_bulletin(*problems: dict, start: str = "2024-02-10T07:00:00Z") -> dict:
    return {
        "bulletinID": "b1",
        "lang": "en",
        "validTime": {"startTime": start},
        "regions": [{"regionID": "CH-1246", "name": "Gadmertal"}],
        "avalancheProblems": list(problems),
    }


def make_problem(**overrides) -> dict:
    problem = {
        "problemType": "wind_slab",
        "dangerRatingValue": "considerable",
        "comment": "Fresh wind slabs.<br/>Caution.",
        "aspects": ["NE", "N"],
        "elevation": {"lowerBound": "2400"},
        "customData": {"CH": {"subdivision": "plus", "coreZoneText": "Level 3+ above 2400m."}},
    }
    return problem | overrides


def make_example(text: str, level: int, valid_from: str) -> dict:
    return {
        "text": text,
        "label": {"problems": [{"danger_level": level}]},
        "valid_from": valid_from,
        "winter": winter(valid_from),
    }


def test_clean_turns_html_into_plain_text():
    assert clean("Wind slabs &amp; new snow. <br/> Be <b>careful</b>.") == (
        "Wind slabs & new snow.\nBe careful."
    )


def test_winter_runs_from_september():
    assert winter("2023-11-28T16:00:00Z") == "2023/24"
    assert winter("2024-05-01T16:00:00Z") == "2023/24"


def test_to_example_converts_labels_and_leaves_out_core_zone_text():
    example = to_example(make_bulletin(make_problem()))

    assert example["winter"] == "2023/24"
    assert "2400m" not in example["text"]
    assert example["label"]["problems"] == [
        {
            "type": "wind_slab",
            "danger_level": 3,
            "danger_modifier": "+",
            "aspects": ["N", "NE"],
            "elevation_lower": 2400,
            "elevation_upper": None,
        }
    ]


def test_to_example_writes_a_shared_description_once():
    gliding = make_problem(problemType="gliding_snow", customData={"CH": {"subdivision": None}})
    example = to_example(make_bulletin(make_problem(), gliding))

    assert example["text"] == "Fresh wind slabs.\nCaution."
    assert example["label"]["problems"][1]["danger_modifier"] is None


def test_to_example_skips_bulletins_listing_a_type_twice():
    assert to_example(make_bulletin(make_problem(), make_problem())) is None


def test_deduplicate_keeps_earliest_and_drops_conflicting_labels():
    examples = [
        make_example("same", 2, "2024-01-02T00:00:00Z"),
        make_example("same", 2, "2024-01-01T00:00:00Z"),
        make_example("conflict", 2, "2024-01-01T00:00:00Z"),
        make_example("conflict", 3, "2024-01-02T00:00:00Z"),
    ]

    assert deduplicate(examples) == [examples[1]]


def test_split_by_winter_and_drops_test_texts_seen_in_training():
    train_example = make_example("old", 2, "2024-01-01T00:00:00Z")
    repeated = make_example("old", 2, "2026-01-01T00:00:00Z")
    new = make_example("new", 2, "2026-01-01T00:00:00Z")

    assert split([train_example, repeated, new]) == ([train_example], [new])
