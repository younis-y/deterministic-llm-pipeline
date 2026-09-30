"""Facts mode end to end: one call per posting, decided by `rules.decide`.

The judge here is a fake that only implements `facts`, never `verdict` or
`triage` - so any test that reaches either of those raises, which is exactly
how a stray call into the judge-mode path would be caught.
"""

from __future__ import annotations

from pathlib import Path

from rolescan.config import LLMConfig, ProfileConfig, RulesConfig
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import (
    FieldFact,
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
    assert cache_key(job, "facts") != cache_key(job, "judge")
    assert cache_key(job, "judge") == job.content_hash
    assert cache_key(job, "facts") == f"{job.content_hash}:facts-v2"


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
        await store.put_verdict(cache_key(job.job, "judge"), stale)

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
        return FitVerdict(fit_score=70, verdict=Verdict.APPLY, confidence="high", reason="ok")

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
        scorer1, judge1 = _facts_scorer(facts, rules=rules, min_report_score=55, store=store)
        [first] = await scorer1.score_all([job])
        assert len(judge1.facts_calls) == 1
        assert first.fit is not None
        assert first.fit.verdict == Verdict.SKIP
        assert first.fit.fit_score <= 54

        # Drop the gate right down. The cached rule-skip must be re-capped
        # under the NEW gate, not replayed with its old score.
        scorer2, judge2 = _facts_scorer(facts, rules=rules, min_report_score=1, store=store)
        [second] = await scorer2.score_all([job])

        assert len(judge2.facts_calls) == 0
        assert second.fit is not None
        assert second.fit.verdict == Verdict.SKIP
        assert second.fit.fit_score == 0, "capped under the new min_report_score of 1"
