"""Fine-tune a small open model with LoRA on n training bulletins and predict the test sample.

python -m avalanche_llm.finetune 16 --smoke      # check the setup in a few minutes
python -m avalanche_llm.finetune 0 100 300 1000

Needs a GPU, e.g. a Kaggle notebook, and pip install -e ".[finetune]". n = 0 scores the
model without fine-tuning. The training loop is written out in plain PyTorch on purpose:
it is short, and it shows what fine-tuning does.
"""

import argparse
import json
import random
import time

from avalanche_llm.config import GPU_USD_PER_HOUR, PREDICTIONS_DIR
from avalanche_llm.dataset import load_split, save
from avalanche_llm.frontier import CALIBRATED

# Text only, Apache 2.0, and small enough to train in fp16 on a free 16 GB T4.
# The newer Qwen3.5-2B is a vision-language model: more machinery for a text task.
MODEL = "Qwen/Qwen3-1.7B"
NAME = "qwen3-1.7b-lora"
# LoRA trains small low-rank updates next to the frozen weights of every linear layer.
LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
EPOCHS = 3
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 2  # Bulletins per forward pass; BATCH_SIZE per optimizer step.
LEARNING_RATE = 2e-4
PREDICT_BATCH_SIZE = 16
MAX_NEW_TOKENS = 512


def training_examples(train: list[dict], n: int, seed: int = 0) -> list[dict]:
    """The first n of one shuffle, so every smaller training set is part of the larger ones."""
    shuffled = train.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


def prompt(tokenizer, text: str) -> str:
    """The same instructions Haiku got. Qwen3 can think first: switched off, as for Haiku's
    short answers, and so the answer starts right after the prompt."""
    messages = [{"role": "system", "content": CALIBRATED}, {"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode(tokenizer, example: dict) -> tuple[list[int], list[int]]:
    """Token ids and labels. The loss covers only the answer: -100 masks the prompt."""
    prompt_ids = tokenizer(prompt(tokenizer, example["text"]))["input_ids"]
    answer = json.dumps(example["label"], ensure_ascii=False) + tokenizer.eos_token
    answer_ids = tokenizer(answer)["input_ids"]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def train(model, tokenizer, examples: list[dict]) -> None:
    import torch

    encoded = [encode(tokenizer, e) for e in examples]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    # A T4 has no bfloat16: compute in float16 and scale the loss so small gradients
    # don't round to zero.
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for epoch in range(EPOCHS):
        random.Random(epoch).shuffle(encoded)
        losses = []
        for batch in batches(encoded, BATCH_SIZE):
            # The logits hold a score for each of Qwen's 152k tokens at every position: for 8
            # bulletins at once they don't fit next to the model. Add up the gradients of
            # smaller pieces instead, weighted so the step is the same as for the whole batch.
            batch_loss = 0.0
            for micro in batches(batch, MICRO_BATCH_SIZE):
                length = max(len(ids) for ids, _ in micro)
                pad = tokenizer.pad_token_id
                input_ids = torch.tensor([ids + [pad] * (length - len(ids)) for ids, _ in micro])
                labels = torch.tensor([lab + [-100] * (length - len(lab)) for _, lab in micro])
                mask = torch.tensor(
                    [[1] * len(ids) + [0] * (length - len(ids)) for ids, _ in micro]
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = (
                        model(
                            input_ids=input_ids.cuda(),
                            attention_mask=mask.cuda(),
                            labels=labels.cuda(),
                        ).loss
                        * len(micro)
                        / len(batch)
                    )
                scaler.scale(loss).backward()
                batch_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            losses.append(batch_loss)
        print(f"epoch {epoch + 1}: loss {sum(losses) / len(losses):.3f}")


def predict(model, tokenizer, examples: list[dict]) -> list[dict]:
    """Greedy answers, with the GPU time per bulletin priced as if the GPU were rented."""
    import torch

    model.eval()
    tokenizer.padding_side = "left"  # Generation continues at the right end of each row.
    predictions = []
    start = time.perf_counter()
    for batch in batches(examples, PREDICT_BATCH_SIZE):
        inputs = tokenizer(
            [prompt(tokenizer, e["text"]) for e in batch], return_tensors="pt", padding=True
        ).to("cuda")
        with torch.no_grad():
            # Training switches the cache off (it clashes with gradient checkpointing). Turn it
            # back on: without it, every new token recomputes the whole text.
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True
            )
        answers = tokenizer.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions += [
            {"id": e["id"], "lang": e["lang"], "output": answer.strip()}
            for e, answer in zip(batch, answers, strict=True)
        ]
    seconds = (time.perf_counter() - start) / len(examples)
    for prediction in predictions:
        prediction["seconds"] = seconds
        prediction["cost_usd"] = seconds * GPU_USD_PER_HOUR / 3600
    return predictions


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("n", type=int, nargs="+", help="numbers of training bulletins")
    parser.add_argument("--smoke", action="store_true", help="16 long bulletins, nothing saved")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_split, test = load_split("train"), load_split("test_sample")
    if args.smoke:
        # The longest bulletins: if memory runs out, it runs out here, not after an hour.
        train_split = sorted(train_split, key=lambda e: len(e["text"]))[-16:]
        test = test[:16]
    for n in args.n:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda()
        if n:
            model.gradient_checkpointing_enable()  # Recompute activations: less memory.
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
            # Only the small LoRA weights are trained: keep them in float32 for stable updates.
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
            tokenizer.padding_side = "right"
            train(model, tokenizer, training_examples(train_split, n))
        predictions = predict(model, tokenizer, test)
        if args.smoke:
            print(predictions[0]["output"], f"{predictions[0]['seconds']:.2f} s per bulletin")
        else:
            save(predictions, PREDICTIONS_DIR / "test_sample" / f"{NAME}-{n}.jsonl")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
