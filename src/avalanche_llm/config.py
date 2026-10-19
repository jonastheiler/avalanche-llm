from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PREDICTIONS_DIR = DATA_DIR / "predictions"

# Train on the first winters, test on the latest one, which no model has seen.
TEST_WINTER = "2025/26"
# LLM calls cost money, so every approach is scored on the same sample of the test set:
# the same bulletins in each language, so languages can be compared fairly.
TEST_SAMPLE_PER_LANGUAGE = 300

# US$ per million input and output tokens. The Batch API halves them.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}
BATCH_DISCOUNT = 0.5

# SLF publishes every bulletin in these four languages, with the same bulletin IDs.
LANGUAGES = ["de", "fr", "it", "en"]
