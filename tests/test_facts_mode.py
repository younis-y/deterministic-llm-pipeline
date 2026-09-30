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
    store: Store | None = None,
    extra_prompt: str = "",
) -> tuple[FitScorer, _FakeFactsJudge]:
    cfg = LLMConfig(enabled=True, backend="ollama", mode="facts")
    scorer = FitScorer(
        cfg, ProfileConfig(rules=rules), store, extra_prompt=extra_prompt
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
    assert cache_key(job, "facts") == f"{job.content_hash}:facts-v1"


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
