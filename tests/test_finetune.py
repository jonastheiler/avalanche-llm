from avalanche_llm.finetune import batches, encode, training_examples


class FakeTokenizer:
    """One token per character, enough to check which tokens the loss sees."""

    eos_token = "$"

    def apply_chat_template(self, messages, **kwargs) -> str:
        return f"<{messages[-1]['content']}>"

    def __call__(self, text: str) -> dict:
        return {"input_ids": [ord(c) for c in text]}


def test_training_examples_are_nested():
    train = [{"id": i} for i in range(50)]

    assert training_examples(train, 30)[:10] == training_examples(train, 10)


def test_encode_masks_the_prompt_and_ends_the_answer():
    ids, labels = encode(FakeTokenizer(), {"text": "ab", "label": {"problems": []}})

    answer = '{"problems": []}$'
    assert ids == [ord(c) for c in "<ab>" + answer]
    assert labels == [-100] * 4 + [ord(c) for c in answer]


def test_batches_keep_the_rest():
    assert batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
