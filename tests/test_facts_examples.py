"""2.5.0: worked examples in the facts system prompt (`llm.facts_examples_file`).

The examples are rendered into the SYSTEM prompt, after the instructions, so
they sit inside the prefix the Anthropic backend caches. The file is the
caller's own (the core carries no example data), validated at load so a bad
example fails the run before any posting is scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rolescan.config import Config, LLMConfig, ProfileConfig
from rolescan.models import Job, ScoredJob
from rolescan.scoring import FitScorer
from rolescan.scoring.examples import (
    FactsExample,
    load_facts_examples,
    render_facts_examples,
)
from rolescan.scoring.facts import PostingFacts
from rolescan.scoring.judges import Judge
from rolescan.scoring.llm import SYSTEM_FACTS, cache_key

EXAMPLES_YAML = """\
- title: Data Science Intern (Summer 2027)
  company: Example Co
  description: |
    A ten-week internship for students graduating in 2028.
    Python and SQL required.
  facts:
    level: {value: graduate_entry, quote: Data Science Intern}
    student_only: {value: true, quote: for students graduating in 2028}
    graduation_year: {value: 2028, quote: graduating in 2028}
    field: {value: data_science, quote: Data Science Intern}
    fit_score: 70
    reason: Python and SQL match; the domain is a fit.
    keywords_missing: []
- title: Analytics Engineer
  company: Other Co
  description: Build dbt models. 3+ years of experience required.
  facts:
    years_required: {value: 3, quote: 3+ years of experience required}
    field: {value: data_engineering, quote: Analytics Engineer}
    fit_score: 60
    reason: dbt overlaps.
"""


def _write(tmp_path: Path, text: str, name: str = "facts_examples.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- loading ---------------------------------------------------------------


def test_examples_load_and_validate_as_posting_facts(tmp_path: Path) -> None:
    examples = load_facts_examples(_write(tmp_path, EXAMPLES_YAML))

    assert [e.title for e in examples] == ["Data Science Intern (Summer 2027)", "Analytics Engineer"]
    assert isinstance(examples[0], FactsExample)
    assert isinstance(examples[0].facts, PostingFacts)
    assert examples[0].facts.graduation_year.value == 2028
    assert examples[1].facts.years_required.value == 3


def test_an_invalid_example_fails_fast_naming_the_example(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace("fit_score: 60", "fit_score: 600")
    path = _write(tmp_path, bad)

    with pytest.raises(ValueError, match=r"example 2 \('Analytics Engineer'\)") as err:
        load_facts_examples(path)
    assert "fit_score" in str(err.value)
    assert str(path) in str(err.value)


def test_an_unknown_fact_key_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace("fit_score: 70", "fit_score: 70\n    seniority: junior")
    with pytest.raises(ValueError, match=r"example 1 \('Data Science Intern \(Summer 2027\)'\)"):
        load_facts_examples(_write(tmp_path, bad))


def test_a_missing_example_field_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace("  company: Other Co\n", "")
    with pytest.raises(ValueError, match="example 2"):
        load_facts_examples(_write(tmp_path, bad))


@pytest.mark.parametrize("text", ["{}\n", "just a string\n", "[]\n", ""])
def test_a_file_that_is_not_a_non_empty_list_fails_fast(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError, match="list"):
        load_facts_examples(_write(tmp_path, text))


def test_a_missing_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"nope\.yaml"):
        load_facts_examples(tmp_path / "nope.yaml")


# --- 2.5.1 review finding M6: every example quote is verbatim -----------------
# An example whose quote is not in its own posting teaches the model to quote
# text that is not there - exactly what the quote guard then throws away.


def test_m6_a_quote_not_in_the_examples_own_posting_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace(
        "quote: 3+ years of experience required", "quote: 5+ years of experience required"
    )
    path = _write(tmp_path, bad)

    with pytest.raises(ValueError, match=r"example 2 \('Analytics Engineer'\)") as err:
        load_facts_examples(path)
    msg = str(err.value)
    assert "years_required" in msg
    assert "5+ years of experience required" in msg
    assert str(path) in msg


def test_m6_a_quote_from_another_examples_posting_fails_fast(tmp_path: Path) -> None:
    """Title and description are the example's OWN - a quote lifted from the
    other example does not verify."""
    bad = EXAMPLES_YAML.replace(
        "field: {value: data_engineering, quote: Analytics Engineer}",
        "field: {value: data_engineering, quote: Python and SQL required}",
    )
    with pytest.raises(ValueError, match=r"example 2 .*Python and SQL required"):
        load_facts_examples(_write(tmp_path, bad))


def test_m6_a_hard_bar_quote_not_in_the_posting_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace(
        "    fit_score: 70\n",
        "    hard_bars: [{kind: clearance, quote: SC clearance required}]\n"
        "    fit_score: 70\n",
    )
    with pytest.raises(ValueError, match=r"example 1 .*hard_bars.*SC clearance required"):
        load_facts_examples(_write(tmp_path, bad))


def test_m6_a_value_with_no_quote_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace(
        "level: {value: graduate_entry, quote: Data Science Intern}",
        "level: {value: graduate_entry, quote: ''}",
    )
    with pytest.raises(ValueError, match=r"example 1 .*level"):
        load_facts_examples(_write(tmp_path, bad))


def test_m6_a_graduation_year_not_in_its_quote_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace(
        "graduation_year: {value: 2028, quote: graduating in 2028}",
        "graduation_year: {value: 2027, quote: graduating in 2028}",
    )
    with pytest.raises(ValueError, match=r"example 1 .*graduation_year"):
        load_facts_examples(_write(tmp_path, bad))


def test_m6_quotes_verify_through_typographic_folding(tmp_path: Path) -> None:
    """The check is the quote guard itself, so a curly apostrophe in the
    posting still matches a straight one in the quote."""
    text = EXAMPLES_YAML.replace(
        "Build dbt models. 3+ years",
        "Build dbt models. You\u2019ll need 3+ years",
    ).replace(
        "quote: 3+ years of experience required",
        "quote: \"You'll need 3+ years\"",
    )
    examples = load_facts_examples(_write(tmp_path, text))
    assert examples[1].facts.years_required.quote == "You'll need 3+ years"


def test_m6_config_load_fails_fast_on_an_unverified_example_quote(tmp_path: Path) -> None:
    _write(tmp_path, EXAMPLES_YAML.replace("quote: Analytics Engineer", "quote: Data Engineer"))
    cfg_path = _write(
        tmp_path, "llm:\n  enabled: false\n  facts_examples_file: facts_examples.yaml\n",
        name="config.yaml",
    )

    with pytest.raises(ValueError, match=r"example 2 \('Analytics Engineer'\).*Data Engineer"):
        Config.load(cfg_path)


# --- rendering -------------------------------------------------------------


def test_examples_render_posting_text_then_expected_json(tmp_path: Path) -> None:
    examples = load_facts_examples(_write(tmp_path, EXAMPLES_YAML))
    text = render_facts_examples(examples)

    assert text.startswith("Worked examples")
    first = text.index("Data Science Intern (Summer 2027)")
    desc = text.index("A ten-week internship for students graduating in 2028.")
    expected = text.index('"graduation_year"')
    second = text.index("Analytics Engineer")
    assert first < desc < expected < second
    assert "Example Co" in text


def test_the_rendered_json_is_the_shape_the_model_returns(tmp_path: Path) -> None:
    [example] = load_facts_examples(_write(tmp_path, EXAMPLES_YAML))[:1]
    text = render_facts_examples([example])
    start = text.index("<expected_facts>") + len("<expected_facts>")
    end = text.index("</expected_facts>")
    payload = json.loads(text[start:end])

    assert PostingFacts.model_validate(payload) == example.facts
    # Provenance is set by code after the call, never by the model.
    assert "source" not in payload["level"]


def test_no_examples_render_nothing() -> None:
    assert render_facts_examples([]) == ""


# --- config ----------------------------------------------------------------


def test_facts_examples_file_defaults_to_none() -> None:
    assert LLMConfig().facts_examples_file is None


def test_config_load_resolves_the_examples_file_against_the_config(tmp_path: Path) -> None:
    _write(tmp_path, EXAMPLES_YAML)
    cfg_path = _write(
        tmp_path, "llm:\n  enabled: false\n  facts_examples_file: facts_examples.yaml\n",
        name="config.yaml",
    )

    cfg = Config.load(cfg_path)

    assert cfg.llm.facts_examples_file == tmp_path / "facts_examples.yaml"


def test_config_load_fails_fast_on_an_invalid_example(tmp_path: Path) -> None:
    _write(tmp_path, EXAMPLES_YAML.replace("fit_score: 60", "fit_score: -1"))
    cfg_path = _write(
        tmp_path, "llm:\n  enabled: false\n  facts_examples_file: facts_examples.yaml\n",
        name="config.yaml",
    )

    with pytest.raises(ValueError, match=r"example 2 \('Analytics Engineer'\)"):
        Config.load(cfg_path)


def test_config_load_fails_fast_on_a_missing_examples_file(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path, "llm:\n  enabled: false\n  facts_examples_file: missing.yaml\n",
        name="config.yaml",
    )
    with pytest.raises(ValueError, match=r"missing\.yaml"):
        Config.load(cfg_path)


# --- the system prompt FitScorer sends -------------------------------------


class _RecordingJudge(Judge):
    name = "recording"
    description = "test only"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self.systems: list[str] = []

    async def verdict(self, system: str, user: str):  # type: ignore[no-untyped-def]
        raise AssertionError("facts mode only")

    async def facts(self, system: str, user: str) -> PostingFacts:
        self.systems.append(system)
        return PostingFacts(fit_score=70, reason="ok")


def _scorer(examples: Path | None) -> tuple[FitScorer, _RecordingJudge]:
    cfg = LLMConfig(enabled=True, backend="ollama", mode="facts", facts_examples_file=examples)
    scorer = FitScorer(cfg, ProfileConfig(summary="A candidate."))
    judge = _RecordingJudge(cfg)
    scorer._judge = judge
    return scorer, judge


def _job() -> ScoredJob:
    return ScoredJob(
        job=Job(source="t", company="Acme", title="Data Engineer", url="https://x/1",
                description="Build pipelines."),
        keyword_score=40,
    )


async def test_examples_render_into_the_system_prompt_after_the_instructions(
    tmp_path: Path,
) -> None:
    scorer, judge = _scorer(_write(tmp_path, EXAMPLES_YAML))

    await scorer.score_all([_job()])

    [system] = judge.systems
    base = SYSTEM_FACTS.format(summary="A candidate.")
    assert system.startswith(base)
    assert "Worked examples" in system[len(base):]
    assert "graduating in 2028" in system
    assert system == scorer.facts_system_prompt()


async def test_no_examples_file_leaves_the_prompt_unchanged() -> None:
    scorer, judge = _scorer(None)

    await scorer.score_all([_job()])

    assert judge.systems == [SYSTEM_FACTS.format(summary="A candidate.")]
    assert scorer.facts_system_prompt() == SYSTEM_FACTS.format(summary="A candidate.")


def test_examples_change_the_facts_cache_key(tmp_path: Path) -> None:
    """Examples change what the model extracts, and the owner edits them, so
    facts cached under one set must not be replayed under another."""
    path = _write(tmp_path, EXAMPLES_YAML)
    with_examples, _ = _scorer(path)
    without, _ = _scorer(None)
    job = _job().job

    base = cache_key(job, "facts", without.cfg)
    plain = without._facts_cache_key(job)
    keyed = with_examples._facts_cache_key(job)
    assert plain.startswith(base + ":") and keyed.startswith(base + ":")
    assert keyed != plain

    path.write_text(EXAMPLES_YAML.replace("fit_score: 60", "fit_score: 61"), encoding="utf-8")
    edited, _ = _scorer(path)
    assert edited._facts_cache_key(job) != keyed


# --- 2.5.2: examples meet the student and hard-bar guards too ---------------


def test_an_example_marking_a_graduates_welcome_advert_student_only_fails_fast(
    tmp_path: Path,
) -> None:
    bad = EXAMPLES_YAML.replace(
        "A ten-week internship for students graduating in 2028.",
        "A ten-week internship for students graduating in 2028. Recent graduates are welcome.",
    )
    with pytest.raises(ValueError, match=r"example 1 .*student_only"):
        load_facts_examples(_write(tmp_path, bad))


def test_an_example_bar_its_quote_does_not_support_fails_fast(tmp_path: Path) -> None:
    bad = EXAMPLES_YAML.replace(
        "    fit_score: 70\n",
        "    hard_bars: [{kind: nationality, quote: Python and SQL required}]\n"
        "    fit_score: 70\n",
    )
    with pytest.raises(ValueError, match=r"example 1 .*hard_bars \(nationality\)"):
        load_facts_examples(_write(tmp_path, bad))
