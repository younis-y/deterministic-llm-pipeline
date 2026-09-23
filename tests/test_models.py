from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from rolescan.models import (
    Confidence,
    CVVariant,
    FitVerdict,
    Job,
    ScoredJob,
    Verdict,
)


def test_uid_is_stable_and_ignores_url(energy_job: Job) -> None:
    """A reposted role with a new requisition id must not read as new."""
    reposted = energy_job.model_copy(update={"url": "https://example.com/99999"})
    assert reposted.uid == energy_job.uid


def test_uid_is_case_and_whitespace_insensitive() -> None:
    a = Job(
        source="s",
        company="Acme",
        title="Data  Scientist",
        location="London",
        url="https://x/1",
    )
    b = Job(
        source="s",
        company="ACME",
        title="data scientist",
        location="london",
        url="https://x/2",
    )
    assert a.uid == b.uid


def test_content_hash_changes_with_description(energy_job: Job) -> None:
    """Editing a posting must invalidate its cached LLM verdict."""
    edited = energy_job.model_copy(update={"description": "different text"})
    assert edited.content_hash != energy_job.content_hash
    assert edited.uid == energy_job.uid


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-08-20", date(2026, 8, 20)),
        ("2026-08-20T11:22:33Z", date(2026, 8, 20)),
        (1755648000000, date.fromtimestamp(1755648000)),  # Lever epoch millis
        ("", None),
        (None, None),
        ("Posted Today", None),  # Workday's non-date
    ],
)
def test_date_coercion(raw: object, expected: date | None) -> None:
    job = Job(source="s", company="c", title="t", url="https://x", posted=raw)
    assert job.posted == expected


def test_job_requires_title_and_url() -> None:
    with pytest.raises(ValidationError):
        Job(source="s", company="c", title="", url="https://x")
    with pytest.raises(ValidationError):
        Job(source="s", company="c", title="t", url="")


def test_whitespace_is_normalised() -> None:
    job = Job(source="s", company="c", title="  Data   \n Scientist ", url="https://x")
    assert job.title == "Data Scientist"


def _verdict(**kw: object) -> FitVerdict:
    base = {
        "fit_score": 80,
        "verdict": Verdict.APPLY,
        "confidence": Confidence.HIGH,
        "reason": "Good match.",
        "cv_variant": CVVariant.ENERGY,
    }
    return FitVerdict.model_validate(base | kw)


def test_llm_score_overrides_keyword_score(energy_job: Job) -> None:
    scored = ScoredJob(job=energy_job, keyword_score=200, fit=_verdict(fit_score=42))
    assert scored.score == 42
    assert scored.verdict is Verdict.APPLY


def test_keyword_score_is_clamped(energy_job: Job) -> None:
    assert ScoredJob(job=energy_job, keyword_score=500).score == 100
    assert ScoredJob(job=energy_job, keyword_score=-80).score == 0


def test_blocked_sinks_below_everything(energy_job: Job, gated_job: Job) -> None:
    good = ScoredJob(job=energy_job, fit=_verdict(fit_score=60))
    blocked = ScoredJob(
        job=gated_job, fit=_verdict(fit_score=95, verdict=Verdict.BLOCKED)
    )
    ranked = sorted([good, blocked], key=lambda s: s.sort_key(), reverse=True)
    assert ranked[0] is good, "a 95-scoring blocked role must not outrank a live one"


def test_fit_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        _verdict(fit_score=101)
    with pytest.raises(ValidationError):
        _verdict(fit_score=-1)


def test_verdict_falls_back_to_keywords(energy_job: Job) -> None:
    assert ScoredJob(job=energy_job, keyword_score=40).verdict is Verdict.CONSIDER
    assert ScoredJob(job=energy_job, keyword_score=0).verdict is Verdict.SKIP
    penalised = ScoredJob(
        job=energy_job, keyword_score=5, keyword_penalties=["uae national"]
    )
    assert penalised.verdict is Verdict.BLOCKED


def test_cv_variants_match_the_files_that_exist() -> None:
    assert {v.value for v in CVVariant} == {
        "CV_AI-LLM-Engineering",
        "CV_Quant-Trading",
        "CV_DataScience-Gulf",
        "CV_EnergySystems-Modelling",
        "CV_Research-DeepLearning",
        "CV_Consulting-Analytics",
    }
