"""The enrichment hook: extra work only for postings that already pass.

All fakes, no network. `_FakeFactsJudge` only implements `facts` - the same
convention as `tests/test_facts_mode.py` - so a stray call into judge mode
would raise rather than pass silently.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from rolescan.config import LLMConfig, ProfileConfig, RulesConfig
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.scoring import enrich as enrich_mod
from rolescan.scoring.enrich import (
    Enricher,
    available_enrichers,
    get_enricher,
    register_enricher,
    unusable_enricher_reason,
)
from rolescan.scoring.facts import (
    FieldFact,
    HardBar,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
)
from rolescan.scoring.judges import Judge
from rolescan.scoring.llm import FitScorer, final_key
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

    name = "fake-facts-enrich"
    description = "test only"
    cheap_triage = True

    def __init__(self, cfg: LLMConfig, facts: PostingFacts) -> None:
        super().__init__(cfg)
        self.facts_calls = 0
        self._facts = facts

    async def verdict(self, system: str, user: str) -> FitVerdict:
        raise AssertionError("facts mode must not call verdict")

    async def triage(self, system: str, user: str) -> FitVerdict:
        raise AssertionError("facts mode must not call triage")

    async def facts(self, system: str, user: str) -> PostingFacts:
        self.facts_calls += 1
        return self._facts


class _ExtraVerdict(FitVerdict):
    """A verdict subclass carrying one field no enricher-unaware caller reads."""

    extra: str = ""


class _FakeEnricher(Enricher):
    name = "fake-enricher"
    verdict_model = _ExtraVerdict

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self.calls: list[Job] = []

    async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        self.calls.append(job)
        return _ExtraVerdict(**verdict.model_dump(), extra="enriched")


class _RaisingEnricher(Enricher):
    name = "raising-enricher"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self.calls = 0

    async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        self.calls += 1
        msg = "enrichment backend is down"
        raise RuntimeError(msg)


class _ConcurrencyTrackingEnricher(Enricher):
    """Records how many `enrich` calls were in flight at once."""

    name = "concurrency-enricher"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return verdict


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the module-level enricher registry per test."""
    monkeypatch.setattr(enrich_mod, "_REGISTRY", {})


def _facts_scorer(
    facts: PostingFacts,
    *,
    enricher_name: str = "",
    min_report_score: int = 55,
    rules: RulesConfig | None = None,
    store: Store | None = None,
) -> tuple[FitScorer, _FakeFactsJudge]:
    cfg = LLMConfig(
        enabled=True, backend="ollama", mode="facts", enricher=enricher_name
    )
    scorer = FitScorer(
        cfg, ProfileConfig(rules=rules, min_report_score=min_report_score), store
    )
    judge = _FakeFactsJudge(cfg, facts)
    scorer._judge = judge
    return scorer, judge


# --- registry ----------------------------------------------------------


def test_get_enricher_returns_none_when_unset() -> None:
    cfg = LLMConfig(enricher="")
    assert get_enricher(cfg) is None


def test_get_enricher_raises_for_an_unknown_name_listing_the_known_set(
    clean_registry: None,
) -> None:
    register_enricher(_FakeEnricher)
    cfg = LLMConfig(enricher="not-a-real-enricher")
    with pytest.raises(ValueError) as e:
        get_enricher(cfg)
    assert "not-a-real-enricher" in str(e.value)
    assert "fake-enricher" in str(e.value)


def test_entry_point_discovery_registers_a_plugin(
    clean_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _PluginEnricher(Enricher):
        name = "plugin-enricher"

        async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
            return verdict

    class _FakeEntryPoint:
        name = "plugin-enricher"

        def load(self) -> type[Enricher]:
            return _PluginEnricher

    monkeypatch.setattr(enrich_mod, "entry_points", lambda **kw: [_FakeEntryPoint()])

    enrich_mod.load_plugins()

    assert available_enrichers()["plugin-enricher"] is _PluginEnricher


# --- the gate: only apply/consider, only above min_report_score --------


@pytest.mark.parametrize(
    ("fit_score", "min_report_score", "hard_bar", "expect_called"),
    [
        pytest.param(80, 55, False, True, id="apply-above-gate"),
        pytest.param(50, 40, False, True, id="consider-above-gate"),
        pytest.param(30, 1, False, False, id="skip-verdict-never-called"),
        pytest.param(90, 1, True, False, id="blocked-verdict-never-called"),
        pytest.param(70, 90, False, False, id="apply-below-gate-not-called"),
    ],
)
async def test_enricher_runs_only_for_apply_or_consider_above_the_gate(
    clean_registry: None,
    fit_score: int,
    min_report_score: int,
    hard_bar: bool,
    expect_called: bool,
) -> None:
    register_enricher(_FakeEnricher)
    if hard_bar:
        facts = _facts(
            fit_score=fit_score,
            hard_bars=[HardBar(kind="clearance", quote="security clearance")],
        )
        description = "Analyst role. Requires an active security clearance."
    else:
        facts = _facts(fit_score=fit_score)
        description = "Analyst role."
    scorer, _judge = _facts_scorer(
        facts, enricher_name="fake-enricher", min_report_score=min_report_score
    )

    [out] = await scorer.score_all([_job(description)])

    enricher = scorer._get_enricher()
    assert isinstance(enricher, _FakeEnricher)
    assert (len(enricher.calls) == 1) is expect_called
    assert scorer.errors == 0
    assert out.fit is not None


# --- failure handling ----------------------------------------------------


async def test_a_raising_enricher_keeps_the_plain_verdict_and_the_posting(
    clean_registry: None,
) -> None:
    register_enricher(_RaisingEnricher)
    facts = _facts(fit_score=80)
    scorer, judge = _facts_scorer(facts, enricher_name="raising-enricher")

    [out] = await scorer.score_all([_job()])

    assert judge.facts_calls == 1
    assert out.fit is not None
    assert out.fit.verdict == Verdict.APPLY
    assert out.fit.fit_score == 80
    assert not isinstance(out.fit, _ExtraVerdict)
    assert scorer.errors == 0, (
        "an enrichment failure must never count as a scoring error"
    )


# --- verdict_model follows the enricher -----------------------------------


async def test_verdict_model_follows_the_configured_enricher(
    clean_registry: None,
) -> None:
    register_enricher(_FakeEnricher)
    facts = _facts(fit_score=80)
    scorer, _judge = _facts_scorer(facts, enricher_name="fake-enricher")

    assert scorer.verdict_model is FitVerdict, "unresolved until the enricher is built"
    await scorer.score_all([_job()])
    assert scorer.verdict_model is _ExtraVerdict


async def test_verdict_model_stays_fitverdict_with_no_enricher_configured() -> None:
    facts = _facts(fit_score=80)
    scorer, _judge = _facts_scorer(facts, enricher_name="")

    await scorer.score_all([_job()])

    assert scorer.verdict_model is FitVerdict


# --- the final verdict cache row ------------------------------------------


async def test_the_enriched_final_verdict_is_written_and_readable_back(
    clean_registry: None, tmp_path: Path
) -> None:
    register_enricher(_FakeEnricher)
    facts = _facts(fit_score=80)

    async with Store(tmp_path / "store.db") as store:
        scorer, judge = _facts_scorer(facts, enricher_name="fake-enricher", store=store)
        job = _job()

        [out] = await scorer.score_all([job])

        assert judge.facts_calls == 1
        assert isinstance(out.fit, _ExtraVerdict)
        assert out.fit.extra == "enriched"

        stored = await store.get_verdict(
            final_key(job.job), scorer.cfg.cache_days, _ExtraVerdict
        )
        assert stored is not None
        assert stored.extra == "enriched"
        assert stored.fit_score == 80


async def test_final_verdict_is_rewritten_on_a_facts_cache_hit(
    clean_registry: None, tmp_path: Path
) -> None:
    """A cache hit skips the model call, but the final verdict must still be
    (re)written - a consumer outside scoring reads `final_key`, never
    `cache_key`, and must see the current decision even when facts extraction
    never ran this time."""
    register_enricher(_FakeEnricher)
    facts = _facts(fit_score=80)
    job = _job()

    async with Store(tmp_path / "store.db") as store:
        scorer, judge = _facts_scorer(facts, enricher_name="fake-enricher", store=store)
        await store.put_verdict(scorer._facts_cache_key(job.job), facts)

        [out] = await scorer.score_all([job])

        assert judge.facts_calls == 0, "facts were already cached"
        assert out.llm_cached is True
        assert isinstance(out.fit, _ExtraVerdict)

        stored = await store.get_verdict(
            final_key(job.job), scorer.cfg.cache_days, _ExtraVerdict
        )
        assert stored is not None
        assert stored.extra == "enriched"


async def test_final_verdict_after_a_raising_enricher_reads_back_as_plain_fitverdict(
    clean_registry: None, tmp_path: Path
) -> None:
    """Minor 4: a failed enrichment writes the PLAIN verdict under `final_key`,
    never a half-built subclass instance."""
    register_enricher(_RaisingEnricher)
    facts = _facts(fit_score=80)

    async with Store(tmp_path / "store.db") as store:
        scorer, _judge = _facts_scorer(
            facts, enricher_name="raising-enricher", store=store
        )
        job = _job()

        [out] = await scorer.score_all([job])

        assert out.fit is not None
        assert type(out.fit) is FitVerdict

        stored = await store.get_verdict(
            final_key(job.job), scorer.cfg.cache_days, FitVerdict
        )
        assert stored is not None
        assert type(stored) is FitVerdict
        assert stored.fit_score == 80
        assert stored.verdict == Verdict.APPLY


# --- concurrency and the call ceiling apply to enrichment too --------------


async def test_enrichment_never_exceeds_max_concurrent(
    clean_registry: None, tmp_path: Path
) -> None:
    register_enricher(_ConcurrencyTrackingEnricher)
    facts = _facts(fit_score=80)
    job1 = _job()
    job2 = ScoredJob(
        job=Job(
            source="test",
            company="Acme",
            title="Data Engineer 2",
            location="London",
            url="https://x/2",
            description="Analyst role.",
        ),
        keyword_score=40,
    )

    async with Store(tmp_path / "store.db") as store:
        cfg = LLMConfig(
            enabled=True,
            backend="ollama",
            mode="facts",
            enricher="concurrency-enricher",
            max_concurrent=1,
        )
        scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), store)
        # Both postings' facts are already cached, so no model call competes
        # for the semaphore - only the two enrichments do.
        await store.put_verdict(scorer._facts_cache_key(job1.job), facts)
        await store.put_verdict(scorer._facts_cache_key(job2.job), facts)

        out = await scorer.score_all([job1, job2])

        enricher = scorer._get_enricher()
        assert isinstance(enricher, _ConcurrencyTrackingEnricher)
        assert enricher.calls == 2
        assert enricher.max_in_flight == 1, (
            "max_concurrent=1 must serialise enrichment the same as any other LLM call"
        )
        assert scorer.errors == 0
        assert all(o.fit is not None for o in out)


async def test_enrichment_is_skipped_once_the_call_ceiling_is_reached(
    clean_registry: None, tmp_path: Path
) -> None:
    register_enricher(_FakeEnricher)
    facts = _facts(fit_score=80)
    job = _job()

    async with Store(tmp_path / "store.db") as store:
        cfg = LLMConfig(
            enabled=True,
            backend="ollama",
            mode="facts",
            enricher="fake-enricher",
            max_calls_per_run=0,
        )
        scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), store)
        await store.put_verdict(scorer._facts_cache_key(job.job), facts)

        [out] = await scorer.score_all([job])

        enricher = scorer._get_enricher()
        assert isinstance(enricher, _FakeEnricher)
        assert enricher.calls == [], "the ceiling was already reached"
        assert out.fit is not None
        assert out.fit.verdict == Verdict.APPLY
        assert not isinstance(out.fit, _ExtraVerdict), "plain verdict kept when skipped"
        assert scorer.errors == 0


# --- an unknown enricher fails fast, not per posting -----------------------


def test_unusable_enricher_reason_is_empty_when_unset() -> None:
    assert unusable_enricher_reason(LLMConfig(enricher="")) == ""


def test_unusable_enricher_reason_is_empty_when_known(clean_registry: None) -> None:
    register_enricher(_FakeEnricher)
    assert unusable_enricher_reason(LLMConfig(enricher="fake-enricher")) == ""


def test_unusable_enricher_reason_names_an_unknown_enricher(
    clean_registry: None,
) -> None:
    register_enricher(_FakeEnricher)
    reason = unusable_enricher_reason(LLMConfig(enricher="not-a-real-enricher"))
    assert "not-a-real-enricher" in reason
    assert "fake-enricher" in reason


async def test_a_scorer_with_an_unknown_enricher_logs_once_and_keeps_plain_verdicts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    facts = _facts(fit_score=80)
    cfg = LLMConfig(
        enabled=True, backend="ollama", mode="facts", enricher="does-not-exist"
    )
    scorer = FitScorer(cfg, ProfileConfig(min_report_score=55))
    scorer._judge = _FakeFactsJudge(cfg, facts)

    jobs = [
        _job(),
        ScoredJob(
            job=Job(
                source="test",
                company="Acme",
                title="Data Engineer 2",
                location="London",
                url="https://x/2",
                description="Analyst role.",
            ),
            keyword_score=40,
        ),
        ScoredJob(
            job=Job(
                source="test",
                company="Acme",
                title="Data Engineer 3",
                location="London",
                url="https://x/3",
                description="Analyst role.",
            ),
            keyword_score=40,
        ),
    ]

    with caplog.at_level(logging.WARNING, logger="rolescan.scoring.llm"):
        out = await scorer.score_all(jobs)

    assert scorer.errors == 0
    for o in out:
        assert o.fit is not None
        assert o.fit.verdict == Verdict.APPLY
        assert type(o.fit) is FitVerdict
    warnings = [r for r in caplog.records if "does-not-exist" in r.getMessage()]
    assert len(warnings) == 1, (
        "an unusable enricher must be logged once, not once per posting"
    )
