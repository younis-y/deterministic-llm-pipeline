from __future__ import annotations

import time
from datetime import date

import pytest
from pydantic import ValidationError

from rolescan.models import (
    Confidence,
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
        (1755648000000, date(2025, 8, 20)),  # Lever epoch millis, read as UTC
        ("", None),
        (None, None),
        ("Posted Today", None),  # Workday's non-date
    ],
)
def test_date_coercion(raw: object, expected: date | None) -> None:
    job = Job(source="s", company="c", title="t", url="https://x", posted=raw)
    assert job.posted == expected


def test_epoch_dates_do_not_move_with_the_machine_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A naive fromtimestamp read Lever's epoch millis against whatever the
    host clock was set to, so a posting published at 23:30 UTC landed on the
    previous day anywhere west of Greenwich. `posted` is what the recency
    window filters on, so the same payload made a role a day older in New
    York than in London."""
    late = 1755647999000  # 2025-08-19T23:59:59Z
    for tz in ("UTC", "America/New_York", "Asia/Dubai"):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        job = Job(source="s", company="c", title="t", url="https://x", posted=late)
        assert job.posted == date(2025, 8, 19), tz


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
        job=energy_job,
        keyword_score=5,
        keyword_penalties=["uae national"],
        blocker_hits=["uae national"],
    )
    assert penalised.verdict is Verdict.BLOCKED


def test_configured_blocker_overrides_a_high_llm_verdict(energy_job: Job) -> None:
    """A hard blocker the user configured is an instruction, not a hint: the
    LLM scoring this posting highly must not be able to outvote it."""
    scored = ScoredJob(
        job=energy_job,
        keyword_penalties=["security clearance"],
        blocker_hits=["security clearance"],
        fit=_verdict(fit_score=88, verdict=Verdict.APPLY),
    )
    assert scored.verdict is Verdict.BLOCKED
    assert scored.is_blocked


def test_a_weighted_term_never_overrides_the_llm_verdict(energy_job: Job) -> None:
    """A term in `blockers` but not in `hard_blockers` (matlab: 15 in the real
    config) is a preference, not an instruction: score_keywords keeps it out
    of blocker_hits, so it must never force `blocked` even though it is still
    recorded in keyword_penalties for display. The LLM's own apply verdict
    stands."""
    scored = ScoredJob(
        job=energy_job,
        keyword_penalties=["matlab"],
        fit=_verdict(fit_score=88, verdict=Verdict.APPLY),
    )
    assert scored.verdict is Verdict.APPLY
    assert not scored.is_blocked


def test_location_mismatch_never_blocks_even_with_a_high_llm_score(
    energy_job: Job,
) -> None:
    """The candidate allows remote and has several target cities: a location
    mismatch is a preference, not a structural bar, and must not become one -
    whether or not the LLM ran."""
    with_llm = ScoredJob(
        job=energy_job,
        keyword_penalties=["location mismatch"],
        fit=_verdict(fit_score=88, verdict=Verdict.APPLY),
    )
    assert with_llm.verdict is Verdict.APPLY
    assert not with_llm.is_blocked

    without_llm = ScoredJob(
        job=energy_job, keyword_score=5, keyword_penalties=["location mismatch"]
    )
    assert without_llm.verdict is Verdict.CONSIDER
    assert not without_llm.is_blocked


def _long_reason(n: int) -> str:
    """`n` characters of plausible prose, never one long word."""
    words = ("strong", "overlap", "with", "the", "day-ahead", "forecasting", "work")
    out: list[str] = []
    while len(" ".join(out)) < n:
        out.append(words[len(out) % len(words)])
    return " ".join(out)[:n]


def test_a_short_reason_is_left_exactly_as_written() -> None:
    assert _verdict(reason="Wrong seniority.").reason == "Wrong seniority."


def test_an_over_long_reason_is_trimmed_not_rejected() -> None:
    """The model does not have to obey the character budget for the verdict to
    be usable, and a verdict already paid for must not be thrown away over
    prose. Ollama reads the schema as a grammar and ignores `maxLength`
    entirely, so this is the only thing holding the limit on that backend."""
    verdict = _verdict(reason=_long_reason(400))
    assert len(verdict.reason) <= 220
    assert verdict.reason.endswith("…"), "a trimmed reason reads as trimmed"
    assert not verdict.reason.endswith(" …"), "cut at a word, not a space"


def test_a_reason_of_exactly_the_limit_is_untouched() -> None:
    text = _long_reason(220)
    assert _verdict(reason=text).reason == text


def test_a_cached_verdict_written_under_the_old_limit_still_loads() -> None:
    """`Store.get_verdict` treats a ValidationError as a schema change and
    deletes the row, so a 400-character reason left in the cache would have
    quietly re-scored every cached posting through the LLM once. It loads."""
    payload = _verdict().model_dump(mode="json") | {"reason": _long_reason(390)}
    restored = FitVerdict.model_validate(payload)
    assert len(restored.reason) <= 220
    assert restored.reason.startswith("strong overlap")


def test_a_non_string_reason_is_still_a_type_error() -> None:
    """Trimming must not turn the field into "accepts anything"."""
    with pytest.raises(ValidationError):
        _verdict(reason=17)


def test_overlong_lists_are_trimmed_not_rejected() -> None:
    # Hosted structured outputs do not enforce maxItems: Haiku returned nine
    # missing keywords on a real posting and the paid-for verdict was lost.
    v = FitVerdict(
        fit_score=70, verdict="consider", confidence="high", reason="Fits.",
        blockers=[f"b{i}" for i in range(7)],
        keywords_missing=[f"k{i}" for i in range(9)],
    )
    assert v.blockers == [f"b{i}" for i in range(5)]
    assert v.keywords_missing == [f"k{i}" for i in range(8)]
