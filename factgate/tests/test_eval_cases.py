import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from factgate.config import Config
from factgate.eval.cases import load_cases
from factgate.eval.grade import grade, offline
from factgate.eval.pytest_helper import assert_case, case_params
from factgate.rules import Decision
from factgate.schema import build_model

CFG = Config.model_validate(
    {
        "name": "t",
        "items": {"jsonl": "x"},
        "facts": {"amount": {"type": "number"}, "late": {"type": "bool"}},
        "rules": [{"name": "big", "when": {"amount": {"gt": 100}}, "then": "review"}],
        "default_outcome": "ok",
        "dangerous": [{"expected": "review", "got": "ok"}],
        "model": {"name": "m"},
    }
)
AS_OF = dt.date(2026, 11, 7)


def _file(tmp_path: Path, name: str, *rows: Any) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


CASE = {"id": "c1", "text": "Total 500", "expect": {"outcome": "review"}}


def test_shorthand_text_and_expect(tmp_path: Path) -> None:
    [case] = load_cases([_file(tmp_path, "a.jsonl", CASE)], CFG)
    assert case.fields == {"text": "Total 500"}
    assert case.expect_outcome == "review" and case.source.endswith("a.jsonl:1")


def test_yaml_cases_load(tmp_path: Path) -> None:
    path = tmp_path / "a.yaml"
    path.write_text("- id: y1\n  text: hi\n  expect: {facts: {amount: {stated: false}}}\n")
    assert load_cases([path], CFG)[0].expect_facts == {"amount": {"stated": False}}


def test_heldout_files_are_marked(tmp_path: Path) -> None:
    cases = load_cases([], CFG, heldout=[_file(tmp_path, "h.jsonl", CASE)])
    assert cases[0].heldout


@pytest.mark.parametrize(
    ("row", "error"),
    [
        ("{not json", r"a\.jsonl:1: not JSON"),
        (
            {"id": "x", "text": "t", "expect": {"facts": {"amount": {"bigger": 1}}}},
            r"a\.jsonl:1: unknown operator 'bigger'",
        ),
        (
            {"id": "x", "text": "t", "expect": {"facts": {"nope": {"eq": 1}}}},
            r"a\.jsonl:1: unknown fact 'nope'",
        ),
        ({"id": "x", "text": "t", "expect": {"outcome": "nope"}}, r"a\.jsonl:1: unknown outcome"),
        ({"id": "x", "text": "t"}, r"a\.jsonl:1: nothing to grade"),
        ({"id": "x", "expect": {"outcome": "ok"}}, r"a\.jsonl:1: no text"),
        (
            {"id": "x", "text": "t", "expect": {"outcome": "ok"}, "offline_only": True},
            r"a\.jsonl:1: an offline_only case needs model_facts",
        ),
        ({"id": "x", "text": "t", "expect": {"outcom": "ok"}}, r"a\.jsonl:1: expect has 'outcom'"),
        (
            {"id": "x", "text": "t", "expect": {"outcome": "ok"}, "model_fact": {}},
            r"a\.jsonl:1: model_fact: Extra inputs are not permitted",
        ),
    ],
)
def test_bad_cases_name_file_and_line(tmp_path: Path, row: Any, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        load_cases([_file(tmp_path, "a.jsonl", row)], CFG)


def test_a_duplicate_id_across_files_names_both(tmp_path: Path) -> None:
    a, b = _file(tmp_path, "a.jsonl", CASE), _file(tmp_path, "b.jsonl", CASE)
    with pytest.raises(ValueError, match=r"duplicate case id 'c1': .*a\.jsonl:1 and .*b\.jsonl:1"):
        load_cases([a, b], CFG)


MODEL = build_model(CFG.facts)


def test_an_outcome_only_failure_and_dangerous(tmp_path: Path) -> None:
    [case] = load_cases(
        [
            _file(
                tmp_path,
                "a.jsonl",
                {**CASE, "expect": {"facts": {"amount": {"gt": 1}}, "outcome": "review"}},
            )
        ],
        CFG,
    )
    facts = MODEL.model_validate({"amount": {"value": 5, "quote": "5"}})
    got = grade(case, facts, Decision(outcome="ok"), CFG, AS_OF)
    assert not got.passed and got.fact_failures == [] and got.outcome_failure == "outcome=ok"
    assert got.dangerous


def test_a_fact_failure_names_the_value(tmp_path: Path) -> None:
    row = {
        "id": "c",
        "text": "Total 500",
        "expect": {"facts": {"amount": {"eq": 500}}},
        "model_facts": {"amount": {"value": 50, "quote": "Total 500"}},
    }
    [case] = load_cases([_file(tmp_path, "a.jsonl", row)], CFG)
    got = offline(case, CFG, MODEL, [], AS_OF)
    assert got.fact_failures == ["amount=None"]  # 50 is not in its quote: cleared


def test_offline_passes_a_good_case(tmp_path: Path) -> None:
    row = {**CASE, "model_facts": {"amount": {"value": 500, "quote": "Total 500"}}}
    [case] = load_cases([_file(tmp_path, "a.jsonl", row)], CFG)
    assert offline(case, CFG, MODEL, [], AS_OF).passed


def test_case_params_mark_open_defects_strict(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        "name: t\nitems: {jsonl: x}\nfacts: {amount: {type: number}}\n"
        "rules: [{name: big, when: {amount: {gt: 100}}, then: review}]\n"
        "default_outcome: ok\nmodel: {name: m}\n"
    )
    good = {**CASE, "model_facts": {"amount": {"value": 500, "quote": "Total 500"}}}
    bad = {**CASE, "id": "c2", "open_defect": "known", "model_facts": {}}
    params = case_params(config, _file(tmp_path, "a.jsonl", good, bad))
    assert [p.id for p in params] == ["c1", "c2"]
    [mark] = params[1].marks
    assert mark.name == "xfail" and mark.kwargs["strict"] is True
    assert_case(params[0].values[0])
    with pytest.raises(AssertionError):
        assert_case(params[1].values[0])


def test_a_yaml_case_names_its_real_line(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "# cases\n"
        "cases:\n"
        "  - id: a\n"
        "    text: Total 500\n"
        "    expect: {outcome: review}\n"
        "\n"
        "  - id: b\n"
        "    text: Total 5\n"
        "    expect: {outcome: nope}\n"
    )
    with pytest.raises(ValueError, match=r"c\.yaml:7: unknown outcome"):
        load_cases([path], CFG)


def test_a_yaml_syntax_error_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("cases: [unclosed")
    with pytest.raises(ValueError, match=r"c\.yaml: not valid YAML"):
        load_cases([path], CFG)
