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
  cur: {type: enum, values: [GBP, EUR]}
  held: {type: bool}
  certs: {type: list, of: text}
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
        ("when: {note: {gt: 10}}", "rule 'big': 'gt' does not apply to text facts"),
        ("when: {amount: {within_days: 3}}", "rule 'big': 'within_days' does not apply"),
        ("when: {note: {matches: '(unclosed'}}", "rule 'big': 'matches' on 'note' is a bad regex"),
        ("when: {}", "rule 'big': `when` is empty"),
        ("when: {amount: {gt: '10,000'}}", "rule 'big': 'gt' on 'amount' needs a number"),
        ("when: {amount: {gt: true}}", "rule 'big': 'gt' on 'amount' needs a number"),
        ("when: {amount: {in: 10}}", "rule 'big': 'in' on 'amount' needs a list"),
        ("when: {due: {before_days: '0'}}", "'before_days' on 'due' needs a whole number"),
        ("when: {due: {gt: next week}}", "'gt' on 'due' needs a date"),
        ("when: {due: {gt: 2026-11-01, in: [2026-11-02, '2026-11-03']}}", None),
        ("when: {note: {matches: 27001}}", "'matches' on 'note' needs text"),
        ("when: {note: {eq: 5}}", "'eq' on 'note' needs text"),
        ("when: {amount: {stated: 'false'}}", "'stated' on 'amount' needs true or false"),
        ("when: {cur: {eq: GPB}}", "'eq' on 'cur' needs one of GBP, EUR"),
        ("when: {cur: {in: [GBP, GPB]}}", "'in' on 'cur' needs one of GBP, EUR"),
        ("when: {held: {is: 'true'}}", "'is' on 'held' needs true or false"),
        ("when: {certs: {contains: 5}}", "'contains' on 'certs' needs text"),
        ("when: {certs: {matches: '27001', stated: true}}", None),
    ],
)
def test_bad_rules_name_the_rule(tmp_path: Path, bad: str, error: str | None) -> None:
    text = GOOD.replace("when: {amount: {gt: 10}}", bad)
    if error is None:
        load_config(_write(tmp_path, text))
        return
    with pytest.raises(ValueError, match=error.replace("(", r"\(")):
        load_config(_write(tmp_path, text))


def test_a_misspelt_fact_key_fails_to_load(tmp_path: Path) -> None:
    """`description:` for `describe:` was ignored, so the model got no description."""
    text = GOOD.replace("amount: {type: number}", "amount: {type: number, description: x}")
    with pytest.raises(ValueError, match=r"facts\.amount\.description"):
        load_config(_write(tmp_path, text))


def test_dangerous_must_name_known_outcomes(tmp_path: Path) -> None:
    text = GOOD + "dangerous: [{expected: review, got: nope}]\n"
    with pytest.raises(ValueError, match="dangerous: 'nope' is not an outcome"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize(
    "bad",
    [
        "model: {name: m, max_concurrent: 0}",
        "model: {name: m, max_calls: -1}",
        "model: {name: m, max_minutes: 0}",
        "model: {name: m, max_tokens: 0}",
    ],
)
def test_model_limits_have_floors(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="greater than"):
        load_config(_write(tmp_path, GOOD.replace("model: {name: m}", bad)))


def test_items_must_name_a_field(tmp_path: Path) -> None:
    text = GOOD.replace("items: {jsonl: items.jsonl}", "items: {jsonl: items.jsonl, fields: []}")
    with pytest.raises(ValueError, match="at least one field"):
        load_config(_write(tmp_path, text))


def test_a_yaml_syntax_error_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"c\.yaml: not valid YAML"):
        load_config(_write(tmp_path, "name: [unclosed"))
