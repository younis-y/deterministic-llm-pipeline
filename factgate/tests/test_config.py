from pathlib import Path

import pytest

from factgate.config import load_config

GOOD = """
name: t
items: {jsonl: items.jsonl}
facts:
  amount: {type: number}
  note: {type: text}
  due: {type: date}
rules:
  - name: big
    when: {amount: {gt: 10}}
    then: review
default_outcome: ok
model: {name: m}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return path


def test_a_good_config_loads_with_its_folder(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, GOOD))
    assert cfg.base == tmp_path
    assert cfg.rules[0].then == "review"
    assert cfg.model.backend == "anthropic"
    assert cfg.cache_path == tmp_path / ".factgate" / "cache.sqlite"


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ("when: {amount: {gt: 10}}", None),
        ("when: {nope: {gt: 10}}", "rule 'big': unknown fact 'nope'"),
        ("when: {amount: {bigger: 10}}", "rule 'big': unknown operator 'bigger'"),
        ("when: {note: {gt: 10}}", "rule 'big': 'gt' does not apply to a text fact"),
        ("when: {amount: {within_days: 3}}", "rule 'big': 'within_days' does not apply"),
        ("when: {note: {matches: '(unclosed'}}", "rule 'big': bad regex"),
        ("when: {}", "rule 'big': `when` is empty"),
    ],
)
def test_bad_rules_name_the_rule(tmp_path: Path, bad: str, error: str | None) -> None:
    text = GOOD.replace("when: {amount: {gt: 10}}", bad)
    if error is None:
        load_config(_write(tmp_path, text))
        return
    with pytest.raises(ValueError, match=error.replace("(", r"\(")):
        load_config(_write(tmp_path, text))


def test_dangerous_must_name_known_outcomes(tmp_path: Path) -> None:
    text = GOOD + "dangerous: [{expected: review, got: nope}]\n"
    with pytest.raises(ValueError, match="dangerous: 'nope' is not an outcome"):
        load_config(_write(tmp_path, text))
