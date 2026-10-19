from types import SimpleNamespace

import pytest

from avalanche_llm.frontier import cost, few_shot_examples, system_prompt, to_prediction


def make_example(lang: str, *types: str) -> dict:
    problems = [{"type": t, "danger_level": 2} for t in types]
    return {"lang": lang, "text": f"{lang} {types}", "label": {"problems": problems}}


def make_result(stop_reason: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1_000_000,
    )
    message = SimpleNamespace(
        stop_reason=stop_reason,
        usage=usage,
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="{}")],
    )
    return SimpleNamespace(type="succeeded", message=message)


def test_few_shot_examples_cover_new_types_in_the_right_language():
    train = [
        make_example("de", "wind_slab"),
        make_example("de", "wind_slab"),
        make_example("de", "gliding_snow"),
        make_example("en", "new_snow"),
    ]
    chosen = few_shot_examples(train, "de", k=5)

    assert sorted(e["label"]["problems"][0]["type"] for e in chosen) == [
        "gliding_snow",
        "wind_slab",
    ]


def test_system_prompt_has_examples_only_when_asked():
    assert "<example>" not in system_prompt([])
    assert "<example>" in system_prompt([make_example("de", "wind_slab")])


def test_cost_counts_cache_reads_at_a_tenth_and_halves_for_batches():
    # Haiku: 1M input at 0.10 + 1M cache reads at 0.01 + 0.1M output at 0.50, halved.
    assert cost(make_result().message.usage, "claude-haiku-5-5") == pytest.approx(0.08)


def test_to_prediction_keeps_text_and_rejects_cut_off_answers():
    custom_id = "de-1234-abcd"

    assert to_prediction(custom_id, make_result(), "claude-haiku-5-5")["output"] == "{}"
    cut_off = to_prediction(custom_id, make_result("max_tokens"), "claude-haiku-5-5")
    assert cut_off["output"] is None
    assert cut_off["id"] == "1234-abcd"
    assert cut_off["cost_usd"] > 0
