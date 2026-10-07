"""Facts mode end to end: one call per posting, decided by `rules.decide`.

The judge here is a fake that only implements `facts`, never `verdict` or
`triage` - so any test that reaches either of those raises, which is exactly
how a stray call into the judge-mode path would be caught.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rolescan.config import LLMConfig, ProfileConfig, RulesConfig
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import (
    FieldFact,
    HardBar,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
)
from rolescan.scoring.judges import Judge
from rolescan.scoring.llm import cache_key
from rolescan.store import Store


def _job(description: str = "Analyst role.") -> ScoredJob:
    job = Job(
        source="test",
        company="Acme",
        title="Data Engineer",
        location="London",
        url="https://x/1",
        description=description,
    )
    return ScoredJob(job=job, keyword_score=40)


def _facts(fit_score: int = 70, **kw: object) -> PostingFacts:
    base: dict[str, object] = {
        "level": LevelFact(),
        "years_required": YearsFact(),
        "student_only": StudentFact(),
        "hard_bars": [],
        "field": FieldFact(),
        "fit_score": fit_score,
        "reason": "Model sentence.",
        "keywords_missing": [],
    }
    base.update(kw)
    return PostingFacts(**base)


class _FakeFactsJudge(Judge):
    """Only implements `facts`. `verdict`/`triage` raise if ever reached."""

    name = "fake-facts"
    description = "test only"
    cheap_triage = True

    def __init__(self, cfg: LLMConfig, facts: PostingFacts) -> None:
        super().__init__(cfg)
        self.facts_calls: list[tuple[str, str]] = []
        self.triage_calls = 0
        self._facts = facts

    async def verdict(self, system: str, user: str) -> FitVerdict:
        raise AssertionError("facts mode must not call verdict")

    async def triage(self, system: str, user: str) -> FitVerdict:
        self.triage_calls += 1
        raise AssertionError("facts mode must not call triage")

    async def facts(self, system: str, user: str) -> PostingFacts:
        self.facts_calls.append((system, user))
        return self._facts


def _facts_scorer(
    facts: PostingFacts,
    *,
    rules: RulesConfig | None = None,
    min_report_score: int = 55,
    store: Store | None = None,
    extra_prompt: str = "",
) -> tuple[FitScorer, _FakeFactsJudge]:
    cfg = LLMConfig(enabled=True, backend="ollama", mode="facts")
    scorer = FitScorer(
        cfg,
        ProfileConfig(rules=rules, min_report_score=min_report_score),
        store,
        extra_prompt=extra_prompt,
    )
    judge = _FakeFactsJudge(cfg, facts)
    scorer._judge = judge
    return scorer, judge


async def test_facts_mode_makes_one_call_and_the_rule_decides() -> None:
    """Years 5 against max_years_required=2 -> skip, quote in the reason."""
    job = _job("Senior role. Must have 5+ years of Python experience.")
    facts = _facts(
        fit_score=80,
        years_required=YearsFact(value=5, quote="5+ years of Python experience"),
        reason="Strong Python overlap.",
    )
    rules = RulesConfig(max_years_required=2)
    scorer, judge = _facts_scorer(facts, rules=rules)

    [out] = await scorer.score_all([job])

    assert len(judge.facts_calls) == 1, "facts mode is one call per posting"
    assert out.fit is not None
    assert out.fit.verdict == Verdict.SKIP
    assert "5+ years of Python experience" in out.fit.reason


async def test_facts_mode_never_triages_even_when_cheap() -> None:
    job = _job("Junior analyst role.")
    facts = _facts(fit_score=70)
    scorer, judge = _facts_scorer(facts)

    [out] = await scorer.score_all([job])

    assert judge.triage_calls == 0
    assert scorer.errors == 0
    assert out.fit is not None


async def test_facts_prompt_carries_no_extra_prompt() -> None:
    job = _job("Analyst role.")
    facts = _facts(fit_score=60)
    scorer, judge = _facts_scorer(facts, extra_prompt="SECRET-BLOCK")

    await scorer.score_all([job])

    system, _user = judge.facts_calls[0]
    assert "SECRET-BLOCK" not in system


async def test_cache_key_is_mode_aware() -> None:
    job = _job().job
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:14b")
    assert cache_key(job, "facts", cfg) != cache_key(job, "judge", cfg)
    assert cache_key(job, "judge", cfg) == job.content_hash
    assert (
        cache_key(job, "facts", cfg)
        == f"{job.content_hash}:facts-v13:ollama:qwen2.5:14b"
    )


async def test_facts_cache_key_changes_with_backend_and_model() -> None:
    """Facts extracted by one model must not be served for 30 days after the
    owner switches to another - the switch the evaluation exists to inform."""
    job = _job().job
    local = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:14b")
    other_model = LLMConfig(enabled=True, backend="ollama", model="llama3.1:8b")
    hosted = LLMConfig(
        enabled=True, backend="anthropic", model="qwen2.5:14b", api_key="k"
    )
    keys = {cache_key(job, "facts", c) for c in (local, other_model, hosted)}
    assert len(keys) == 3
    # judge mode is unchanged: the bare content hash whatever the backend
    assert {cache_key(job, "judge", c) for c in (local, hosted)} == {job.content_hash}


async def test_a_facts_row_cached_under_another_model_is_not_read_back(
    tmp_path: Path,
) -> None:
    job = _job("Analyst role.")
    async with Store(tmp_path / "store.db") as store:
        old = LLMConfig(enabled=True, backend="ollama", model="older-model")
        await store.put_verdict(cache_key(job.job, "facts", old), _facts(fit_score=99))

        scorer, judge = _facts_scorer(_facts(fit_score=60), store=store)
        [out] = await scorer.score_all([job])

        assert len(judge.facts_calls) == 1
        assert out.fit is not None and out.fit.fit_score == 60


async def test_a_judge_mode_cache_entry_is_not_read_back_in_facts_mode(
    tmp_path: Path,
) -> None:
    job = _job("Analyst role.")

    async with Store(tmp_path / "store.db") as store:
        stale = FitVerdict(
            fit_score=90,
            verdict=Verdict.APPLY,
            confidence="high",
            reason="stale judge-mode verdict",
        )
        await store.put_verdict(cache_key(job.job, "judge", LLMConfig()), stale)

        facts = _facts(fit_score=60, reason="fresh facts-mode verdict")
        scorer, judge = _facts_scorer(facts, store=store)

        [out] = await scorer.score_all([job])

        assert len(judge.facts_calls) == 1, (
            "a verdict cached under the judge-mode key must not satisfy a "
            "facts-mode read"
        )
        assert out.fit is not None
        assert out.fit.reason != "stale judge-mode verdict"


class _JudgeModeJudge(Judge):
    """Only implements `verdict`. `facts` raises if ever reached."""

    name = "fake-judge-mode"
    description = "test only"
    cheap_triage = False

    async def verdict(self, system: str, user: str) -> FitVerdict:
        return FitVerdict(
            fit_score=70, verdict=Verdict.APPLY, confidence="high", reason="ok"
        )

    async def facts(self, system: str, user: str) -> PostingFacts:
        raise AssertionError("judge mode must not call facts")


async def test_judge_mode_still_routes_through_verdict_not_facts() -> None:
    cfg = LLMConfig(enabled=True, backend="ollama", mode="judge")
    scorer = FitScorer(cfg, ProfileConfig())
    scorer._judge = _JudgeModeJudge(cfg)

    [out] = await scorer.score_all([_job("Analyst role.")])

    assert out.fit is not None
    assert out.fit.fit_score == 70


# --- the cache holds verified facts, not the post-rules verdict ------------
#
# Two configs can read the same cache row and reach different verdicts, since
# `rules` and `min_report_score` are applied AFTER the cache lookup, not
# baked into what is stored. Caching the final FitVerdict instead would freeze
# whatever rules were active on the first run; caching the facts means a
# rules or min_report_score change takes effect immediately, for every
# posting already in cache, at zero extra model calls.


async def test_a_cached_facts_hit_is_redecided_under_the_current_rules(
    tmp_path: Path,
) -> None:
    job = _job("Senior role. Must have 5+ years of Python experience.")
    facts = _facts(
        fit_score=80,
        years_required=YearsFact(value=5, quote="5+ years of Python experience"),
    )

    async with Store(tmp_path / "store.db") as store:
        strict = RulesConfig(max_years_required=2)
        scorer1, judge1 = _facts_scorer(facts, rules=strict, store=store)
        [first] = await scorer1.score_all([job])
        assert len(judge1.facts_calls) == 1
        assert first.fit is not None and first.fit.verdict == Verdict.SKIP
        assert scorer1.last_facts[job.job.url].years_required.value == 5

        # A second, looser config sharing the same store and cache row.
        loose = RulesConfig(max_years_required=10)
        scorer2, judge2 = _facts_scorer(facts, rules=loose, store=store)
        [second] = await scorer2.score_all([job])

        assert len(judge2.facts_calls) == 0, (
            "the cached facts must satisfy this without a new model call"
        )
        assert second.llm_cached is True
        assert second.fit is not None
        assert second.fit.verdict == Verdict.APPLY, (
            "the years rule no longer fires under the looser config, so the "
            "fit_score band decides: 80 -> apply"
        )
        assert scorer2.last_facts[job.job.url].years_required.value == 5, (
            "a cache hit must still populate last_facts (Minor 1)"
        )


async def test_a_cached_rule_skip_never_reaches_the_digest_after_the_gate_drops(
    tmp_path: Path,
) -> None:
    """A rule skip caps its score just under `min_report_score` so it can
    never look better than a posting that reached the digest on fit alone.
    That cap must be re-applied on every cache hit, using the CURRENT
    min_report_score, or lowering the gate later would let a stale cached
    verdict re-appear unnaturally high."""
    job = _job("Senior role. Must have 5+ years of Python experience.")
    facts = _facts(
        fit_score=80,
        years_required=YearsFact(value=5, quote="5+ years of Python experience"),
    )
    rules = RulesConfig(max_years_required=2)

    async with Store(tmp_path / "store.db") as store:
        scorer1, judge1 = _facts_scorer(
            facts, rules=rules, min_report_score=55, store=store
        )
        [first] = await scorer1.score_all([job])
        assert len(judge1.facts_calls) == 1
        assert first.fit is not None
        assert first.fit.verdict == Verdict.SKIP
        assert first.fit.fit_score <= 54

        # Drop the gate right down. The cached rule-skip must be re-capped
        # under the NEW gate, not replayed with its old score.
        scorer2, judge2 = _facts_scorer(
            facts, rules=rules, min_report_score=1, store=store
        )
        [second] = await scorer2.score_all([job])

        assert len(judge2.facts_calls) == 0
        assert second.fit is not None
        assert second.fit.verdict == Verdict.SKIP
        assert second.fit.fit_score == 0, "capped under the new min_report_score of 1"


# --- the deterministic level pass runs on the live path (fix A) ------------


async def test_a_senior_title_the_model_left_not_stated_skips_with_the_title_quote() -> (
    None
):
    job = ScoredJob(
        job=Job(
            source="test",
            company="Harnham",
            title="Senior AI Engineer (198174)",
            url="https://x/9",
            description="Build agentic systems.",
        ),
        keyword_score=40,
    )
    facts = _facts(fit_score=75, level=LevelFact(value="not_stated", quote=""))
    rules = RulesConfig(allowed_levels=["graduate_entry", "junior", "not_stated"])
    scorer, _judge = _facts_scorer(facts, rules=rules)

    [out] = await scorer.score_all([job])

    assert out.fit is not None
    assert out.fit.verdict == Verdict.SKIP
    assert out.fit.reason == 'Skip: advert is for "Senior AI Engineer (198174)"'
    assert scorer.last_facts[job.job.url].level.value == "senior"


async def test_a_graduate_title_still_passes_on_fit() -> None:
    job = ScoredJob(
        job=Job(
            source="test",
            company="FDM",
            title="Graduate AI Engineer",
            url="https://x/10",
            description="Graduate programme.",
        ),
        keyword_score=40,
    )
    rules = RulesConfig(allowed_levels=["graduate_entry", "junior", "not_stated"])
    scorer, _judge = _facts_scorer(_facts(fit_score=75), rules=rules)

    [out] = await scorer.score_all([job])

    assert out.fit is not None and out.fit.verdict == Verdict.APPLY


# --- the deterministic field pass runs on the live path (2.4.3) -----------


_OWNER_FIELDS = RulesConfig(
    allowed_fields=["data_engineering", "ai_llm", "data_science"]
)


async def test_an_analyst_title_the_model_left_fieldless_skips_with_the_title_quote() -> (
    None
):
    job = ScoredJob(
        job=Job(
            source="test",
            company="Acme",
            title="Graduate Data Analyst",
            url="https://x/11",
            description="Dashboards for the sales team.",
        ),
        keyword_score=40,
    )
    scorer, _judge = _facts_scorer(_facts(fit_score=75), rules=_OWNER_FIELDS)

    [out] = await scorer.score_all([job])

    assert out.fit is not None
    assert out.fit.verdict == Verdict.SKIP
    assert out.fit.reason == 'Skip: advert is for "Graduate Data Analyst"'
    assert scorer.last_facts[job.job.url].field.value == "analytics_bi"


async def test_a_data_engineer_title_passes_the_field_rule_on_fit() -> None:
    scorer, _judge = _facts_scorer(_facts(fit_score=75), rules=_OWNER_FIELDS)

    [out] = await scorer.score_all([_job("Build pipelines.")])

    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    assert scorer.last_facts["https://x/1"].field.value == "data_engineering"


async def test_cached_facts_hold_the_resolved_field(tmp_path: Path) -> None:
    job = _job("Build pipelines.")
    async with Store(tmp_path / "store.db") as store:
        scorer, _judge = _facts_scorer(_facts(fit_score=75), store=store)
        await scorer.score_all([job])
        key = cache_key(job.job, "facts", scorer.cfg)
        cached = await store.get_verdict(key, 30, PostingFacts)

    assert cached is not None
    assert cached.field.value == "data_engineering"
    assert cached.field.quote == "Data Engineer"


# --- the field-exempt company reaches `decide` on both scorer paths (2.5.7) --


_EXEMPT = RulesConfig(
    allowed_fields=["data_science"], field_exempt_companies=["Example Bank"]
)


def _software_job(company: str) -> ScoredJob:
    """A graduate software role: hidden by the field rule unless exempt."""
    return ScoredJob(
        job=Job(
            source="test",
            company=company,
            title="Graduate Software Engineer",
            url="https://x/12",
            description="Graduate programme.",
        ),
        keyword_score=40,
    )


async def test_an_exempt_company_passes_the_field_rule_on_a_fresh_call() -> None:
    scorer, _judge = _facts_scorer(_facts(fit_score=75), rules=_EXEMPT)
    [out] = await scorer.score_all([_software_job("Example Bank Ltd")])
    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    assert out.fit.rule is None

    scorer, _judge = _facts_scorer(_facts(fit_score=75), rules=_EXEMPT)
    [other] = await scorer.score_all([_software_job("Other Co")])
    assert other.fit is not None and other.fit.verdict == Verdict.SKIP
    assert other.fit.rule == "field"


async def test_an_exempt_company_passes_the_field_rule_on_a_cached_hit(
    tmp_path: Path,
) -> None:
    job = _software_job("Example Bank Ltd")
    async with Store(tmp_path / "store.db") as store:
        scorer1, judge1 = _facts_scorer(
            _facts(fit_score=75), rules=_EXEMPT, store=store
        )
        await scorer1.score_all([job])
        assert len(judge1.facts_calls) == 1

        scorer2, judge2 = _facts_scorer(
            _facts(fit_score=75), rules=_EXEMPT, store=store
        )
        [out] = await scorer2.score_all([job])

    assert len(judge2.facts_calls) == 0, "this must be the cached-facts path"
    assert out.llm_cached is True
    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    assert out.fit.rule is None


# --- a nationality bar the candidate meets is not a bar (2.5.7) -----


def _job_with_nationality_bar(
    quote: str = "Jordanian nationals only",
) -> tuple[ScoredJob, HardBar]:
    """A data engineer role with a nationality bar."""
    return (
        ScoredJob(
            job=Job(
                source="test",
                company="Amman Tech",
                title="Data Engineer",
                url="https://x/15",
                description=f"{quote}. Work with our team.",
            ),
            keyword_score=40,
        ),
        HardBar(kind="nationality", quote=quote),
    )


async def test_a_nationality_bar_the_candidate_meets_passes_on_a_fresh_call() -> None:
    job, bar = _job_with_nationality_bar()
    facts = _facts(fit_score=75, hard_bars=[bar])
    scorer = FitScorer(
        LLMConfig(enabled=True, backend="ollama", mode="facts"),
        ProfileConfig(nationalities=["jordanian", "jordan"]),
    )
    judge = _FakeFactsJudge(scorer.cfg, facts)
    scorer._judge = judge
    [out] = await scorer.score_all([job])
    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    assert out.fit.rule is None


async def test_a_nationality_bar_the_candidate_meets_passes_on_a_cached_hit(
    tmp_path: Path,
) -> None:
    job, bar = _job_with_nationality_bar()
    facts = _facts(fit_score=75, hard_bars=[bar])

    async with Store(tmp_path / "store.db") as store:
        scorer1 = FitScorer(
            LLMConfig(enabled=True, backend="ollama", mode="facts"),
            ProfileConfig(nationalities=["jordanian"]),
            store,
        )
        judge1 = _FakeFactsJudge(scorer1.cfg, facts)
        scorer1._judge = judge1
        await scorer1.score_all([job])
        assert len(judge1.facts_calls) == 1

        scorer2 = FitScorer(
            LLMConfig(enabled=True, backend="ollama", mode="facts"),
            ProfileConfig(nationalities=["jordanian"]),
            store,
        )
        judge2 = _FakeFactsJudge(scorer2.cfg, facts)
        scorer2._judge = judge2
        [out] = await scorer2.score_all([job])

    assert len(judge2.facts_calls) == 0, "this must be the cached-facts path"
    assert out.llm_cached is True
    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    assert out.fit.rule is None


# --- extra_prompt is not sent in facts mode; say so once (fix E) -----------


def test_facts_mode_with_extra_prompt_and_no_enricher_logs_once_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="rolescan.scoring.llm"):
        FitScorer(
            LLMConfig(enabled=True, backend="ollama", mode="facts"),
            ProfileConfig(),
            extra_prompt="private block",
        )
    hits = [r for r in caplog.records if "extra_prompt" in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelno == logging.INFO


@pytest.mark.parametrize(
    ("mode", "enricher", "extra"),
    [
        ("judge", "", "private"),
        ("facts", "some-enricher", "private"),
        ("facts", "", ""),
    ],
)
def test_no_extra_prompt_notice_otherwise(
    caplog: pytest.LogCaptureFixture, mode: str, enricher: str, extra: str
) -> None:
    with caplog.at_level(logging.INFO, logger="rolescan.scoring.llm"):
        FitScorer(
            LLMConfig(enabled=True, backend="ollama", mode=mode, enricher=enricher),
            ProfileConfig(),
            extra_prompt=extra,
        )
    assert not [r for r in caplog.records if "extra_prompt" in r.getMessage()]


# --- prompt wording (fixes B and C) ----------------------------------------


def test_facts_prompt_asks_for_short_quotes() -> None:
    from rolescan.scoring.llm import SYSTEM_FACTS

    assert "under 200 characters" in SYSTEM_FACTS


def test_facts_prompt_narrows_the_other_bar() -> None:
    from rolescan.scoring.llm import SYSTEM_FACTS

    text = " ".join(SYSTEM_FACTS.split())
    assert "other: an explicit, mandatory eligibility requirement" in text
    assert "never experience, sector background or skills" in text


def test_facts_prompt_says_work_auth_is_checked_by_keywords() -> None:
    """2.4.2: work_auth bars are still extracted (the eval measures them) but
    decide() ignores them; the prompt must not pretend the model decides."""
    from rolescan.scoring.llm import SYSTEM_FACTS

    text = " ".join(SYSTEM_FACTS.split())
    assert "- work_auth:" in text
    assert "work authorisation is checked by configured keywords" in text
    assert "not by your judgement of the candidate" in text


# --- 2.5.2: the recent-graduates guard runs on the live path ---------------


async def test_a_student_flag_on_an_advert_that_accepts_graduates_does_not_skip() -> (
    None
):
    job = _job("Requirements: Recent graduates or final year students.")
    rules = RulesConfig(student_only="skip")
    scorer, _judge = _facts_scorer(
        _facts(
            fit_score=75,
            student_only=StudentFact(value=True, quote="final year students"),
        ),
        rules=rules,
    )

    [out] = await scorer.score_all([job])

    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    student = scorer.last_facts[job.job.url].student_only
    assert student.value is False
    assert student.quote == "Requirements: Recent graduates or final year students."


async def test_an_unsupported_clearance_bar_does_not_block_on_the_live_path() -> None:
    from rolescan.models import BarKind
    from rolescan.scoring.facts import HardBar

    job = _job("Adheres to the established internal security practices.")
    scorer, _judge = _facts_scorer(
        _facts(
            fit_score=75,
            hard_bars=[
                HardBar(kind=BarKind.clearance, quote="internal security practices"),
            ],
        ),
    )

    [out] = await scorer.score_all([job])

    assert out.fit is not None and out.fit.verdict == Verdict.APPLY
    assert scorer.last_facts[job.job.url].hard_bars == []


async def test_a_bar_the_model_left_out_still_blocks_through_the_scorer() -> None:
    """2.5.6: `_call_facts` chains `resolve_hard_bars` after `verify_facts`."""
    job = _job("Motivated and hardworking. This role is open to UAE Nationals only.")
    scorer, _judge = _facts_scorer(_facts(fit_score=85))

    [out] = await scorer.score_all([job])

    assert out.fit is not None
    assert out.fit.verdict == Verdict.BLOCKED
    assert "open to UAE Nationals only" in out.fit.reason
