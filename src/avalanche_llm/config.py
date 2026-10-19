from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Train on the first winters, test on the latest one, which no model has seen.
TEST_WINTER = "2025/26"

# SLF publishes every bulletin in these four languages, with the same bulletin IDs.
LANGUAGES = ["de", "fr", "it", "en"]
