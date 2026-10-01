"""2.5.0: `rules.level_from_title_only` and the provenance it reads.

The owner judges seniority from the job title. A model that reads "mid" or
"senior" out of the description (a sentence about the team, a "we value
experience" line) skipped good roles, so with the flag on only a level read
from the title can fire the level rule.
"""

from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import Job, JobField, Level
from rolescan.scoring.facts import (
    FieldFact,
    LevelFact,
    PostingFacts,
    resolve_level,
    verify_facts,
)
from rolescan.scoring.rules import decide

GRAD_RULES = RulesConfig(allowed_levels=[Level.graduate_entry, Level.junior, Level.not_stated])
TITLE_ONLY = GRAD_RULES.model_copy(update={"level_from_title_only": True})


def _job(title: str, description: str) -> Job:
    return Job(source="t", company="Acme", title=title, url="https://x/1",
               description=description)


def _facts(level: LevelFact, fit: int = 80) -> PostingFacts:
    return PostingFacts(
        level=level,
        field=FieldFact(value=JobField.data_engineering, quote="Data Engineer"),
        fit_score=fit,
        reason="Model sentence.",
    )


def _resolved(job: Job, level: LevelFact) -> PostingFacts:
    return resolve_level(verify_facts(_facts(level), job), job)


# --- provenance ------------------------------------------------------------


def test_the_flag_defaults_off() -> None:
    assert RulesConfig().level_from_title_only is False


def test_a_level_fact_defaults_to_no_source() -> None:
    assert LevelFact().source == "none"


def test_provenance_is_not_in_the_schema_the_model_fills() -> None:
    """The model never states where a level came from; `resolve_level` does,
    after the quote guard. Asking the model would let it claim "title"."""
    props = PostingFacts.model_json_schema()["$defs"]["LevelFact"]["properties"]
    assert "source" not in props


def test_a_title_keyword_level_is_sourced_from_the_title() -> None:
    got = _resolved(_job("Senior Data Engineer", "Build pipelines."), LevelFact())
    assert got.level.value == Level.senior and got.level.source == "title"


def test_a_model_level_quoted_from_the_description_is_sourced_from_text() -> None:
    job = _job("Data Engineer", "This is a mid-level role in a small team.")
    got = _resolved(job, LevelFact(value=Level.mid, quote="mid-level role"))
    assert got.level.value == Level.mid and got.level.source == "text"


def test_a_level_derived_from_a_description_quote_is_sourced_from_text() -> None:
    job = _job("Data Engineer", "Hiring a senior engineer to own the platform.")
    got = _resolved(job, LevelFact(value=Level.not_stated, quote="a senior engineer"))
    assert got.level.value == Level.senior and got.level.source == "text"


def test_a_model_level_quoted_from_the_title_is_sourced_from_the_title() -> None:
    """No level keyword in the title, but the model read "mid" from the title
    itself - that is still the title speaking."""
    job = _job("Mid-Level Data Engineer", "Build pipelines.")
    got = _resolved(job, LevelFact(value=Level.mid, quote="Mid-Level Data Engineer"))
    assert got.level.value == Level.mid and got.level.source == "title"


def test_no_level_has_no_source() -> None:
    got = _resolved(_job("Data Engineer", "Build pipelines."), LevelFact())
    assert got.level.value == Level.not_stated and got.level.source == "none"


def test_a_model_claimed_source_is_overwritten_by_resolve_level() -> None:
    job = _job("Data Engineer", "This is a mid-level role in a small team.")
    got = _resolved(job, LevelFact(value=Level.mid, quote="mid-level role", source="title"))
    assert got.level.source == "text"


def test_resolve_level_stays_idempotent_with_provenance() -> None:
    job = _job("Data Engineer", "This is a mid-level role in a small team.")
    once = _resolved(job, LevelFact(value=Level.mid, quote="mid-level role"))
    assert resolve_level(once, job) == once


# --- the rule --------------------------------------------------------------


@pytest.mark.parametrize("level", [Level.mid, Level.senior])
def test_flag_on_a_description_level_does_not_skip(level: Level) -> None:
    job = _job("Data Engineer", f"We want a {level.value} engineer for this team.")
    facts = _resolved(job, LevelFact(value=level, quote=f"a {level.value} engineer"))
    assert facts.level.source == "text"

    v = decide(facts, TITLE_ONLY, 55)

    assert v.verdict.value == "apply"
    assert v.reason == "Model sentence."


def test_flag_on_a_title_level_still_skips() -> None:
    facts = _resolved(_job("Senior Data Engineer", "Build pipelines."), LevelFact())

    v = decide(facts, TITLE_ONLY, 55)

    assert v.verdict.value == "skip"
    assert v.reason == 'Skip: advert is for "Senior Data Engineer"'


@pytest.mark.parametrize("level", [Level.mid, Level.senior])
def test_flag_off_a_description_level_skips_as_before(level: Level) -> None:
    job = _job("Data Engineer", f"We want a {level.value} engineer for this team.")
    facts = _resolved(job, LevelFact(value=level, quote=f"a {level.value} engineer"))

    v = decide(facts, GRAD_RULES, 55)

    assert v.verdict.value == "skip"
    assert v.reason == f'Skip: advert is for "a {level.value} engineer"'


def test_flag_off_an_unresolved_level_fact_still_fires() -> None:
    """A caller that builds facts by hand (no `resolve_level`, source "none")
    gets exactly the 2.4.4 behaviour with the flag off."""
    facts = _facts(LevelFact(value=Level.senior, quote="Senior Data Engineer"))
    assert decide(facts, GRAD_RULES, 55).verdict.value == "skip"


def test_flag_on_an_unresolved_level_fact_does_not_fire() -> None:
    facts = _facts(LevelFact(value=Level.senior, quote="Senior Data Engineer"))
    assert decide(facts, TITLE_ONLY, 55).verdict.value == "apply"


def test_flag_on_leaves_the_other_rules_alone() -> None:
    from rolescan.scoring.facts import YearsFact

    job = _job("Data Engineer", "A senior engineer with 5+ years of Python.")
    facts = _resolved(job, LevelFact(value=Level.senior, quote="A senior engineer"))
    facts = facts.model_copy(
        update={"years_required": YearsFact(value=5, quote="5+ years of Python")}
    )
    rules = TITLE_ONLY.model_copy(update={"max_years_required": 2})

    v = decide(facts, rules, 55)

    assert v.verdict.value == "skip"
    assert "5+ years of Python" in v.reason
