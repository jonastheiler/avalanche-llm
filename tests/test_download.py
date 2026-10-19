from avalanche_llm.download import unique_issues


def make_issue(*bulletin_ids: str) -> dict:
    return {"bulletins": [{"bulletinID": b} for b in bulletin_ids]}


def test_unique_issues_drops_repeats_and_keeps_order():
    issues = [make_issue("a", "b"), make_issue("c"), make_issue("b", "a")]

    assert unique_issues(issues) == [make_issue("a", "b"), make_issue("c")]


def test_unique_issues_drops_empty_issues():
    assert unique_issues([make_issue()]) == []
