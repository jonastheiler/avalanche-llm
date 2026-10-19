"""Download the archive of Swiss avalanche bulletins from the SLF API.

python -m avalanche_llm.download            # all languages
python -m avalanche_llm.download de en      # some languages

Data: WSL Institute for Snow and Avalanche Research SLF, www.slf.ch, CC BY 4.0.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from avalanche_llm.config import LANGUAGES, RAW_DIR

# Every item in the list is one issue (morning or evening), holding one bulletin per group
# of warning regions that share the same danger. The API archive starts in November 2023.
LIST_URL = "https://aws.slf.ch/api/bulletin-list/caaml/{lang}/json"
PAGE_SIZE = 50
# The SLF asks users not to overload its servers, so we pause between pages.
PAUSE_SECONDS = 1

RETRY = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=RETRY))


def fetch_page(lang: str, offset: int) -> list[dict]:
    params = {"limit": PAGE_SIZE, "offset": offset}
    response = HTTP.get(LIST_URL.format(lang=lang), params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def issue_key(issue: dict) -> tuple[str, ...]:
    return tuple(sorted(b["bulletinID"] for b in issue["bulletins"]))


def unique_issues(issues: list[dict]) -> list[dict]:
    """Drop repeated issues. Paging runs newest first, so if SLF publishes a new issue
    while we download, every later page shifts by one and repeats an issue."""
    seen, unique = set(), []
    for issue in issues:
        if issue["bulletins"] and issue_key(issue) not in seen:
            seen.add(issue_key(issue))
            unique.append(issue)
    return unique


def download(lang: str) -> list[dict]:
    issues, offset = [], 0
    while page := fetch_page(lang, offset):
        issues += page
        offset += len(page)
        time.sleep(PAUSE_SECONDS)
    return unique_issues(issues)


def save(issues: list[dict], path: Path) -> None:
    # Gzipped JSON keeps the raw API response unchanged at about 5% of its size,
    # small enough to commit, so results stay reproducible if the API changes.
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False)


def load(lang: str) -> list[dict]:
    with gzip.open(RAW_DIR / f"{lang}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("languages", nargs="*", choices=LANGUAGES)
    for lang in parser.parse_args().languages or LANGUAGES:
        issues = download(lang)
        save(issues, RAW_DIR / f"{lang}.json.gz")
        starts = [b["validTime"]["startTime"] for i in issues for b in i["bulletins"]]
        print(f"{lang}: {len(issues)} issues, {min(starts)[:10]} to {max(starts)[:10]}")


if __name__ == "__main__":
    main()
