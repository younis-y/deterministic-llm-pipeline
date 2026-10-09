"""What the model did each run, kept in `llm_runs`, and the digest line that
says when it changed (2.6.0).

Four parts: the table and its migration, the counters `finish_facts` keeps,
the scan that records one row, and the "Model health" line. The line is an
alarm in the digest's opening block: it speaks only when a rate is more than
twice what the last five runs show, and only with three earlier runs to
compare against.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import rolescan.store as store_module
from rolescan.config import LLMConfig, ProfileConfig
from rolescan.health import HealthFlag, model_health
from rolescan.models import BarKind, FitVerdict, Job, JobField, Level, ScoredJob
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import (
    FieldFact,
    HardBar,
    LevelFact,
    PostingFacts,
    YearsFact,
)
from rolescan.scoring.judges import Judge
from rolescan.scoring.llm import FactsCounter, finish_facts
from rolescan.store import _MIGRATIONS, LlmRun, Store, StoreTooNewError

# --- the table and its migration ---------------------------------------------

_COLUMNS = [
    "run_id",
    "ran",
    "backend",
    "model",
    "model_digest",
    "calls",
    "cached",
    "errors",
    "breaker",
    "deferred",
    "quotes_rejected",
    "level_overridden",
    "field_overridden",
    "years_set",
    "years_cleared",
    "bars_added",
    "postings",
]


def _table(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as conn:
        return [r[1] for r in conn.execute("PRAGMA table_info(llm_runs)")]


def _version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


async def test_a_new_store_has_the_llm_runs_table(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    async with Store(path):
        pass

    assert _table(path) == _COLUMNS
    assert _version(path) == len(_MIGRATIONS) == 10


async def test_migration_9_adds_the_table_to_a_store_at_version_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.db"
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", _MIGRATIONS[:8])
        async with Store(path) as store:
            await store.db.execute(
                "INSERT INTO source_counts (source_key, ran, count) "
                "VALUES ('k', '2026-01-01T00:00:00+00:00', 3)"
            )
            await store.db.commit()
    assert _table(path) == []
    assert _version(path) == 8

    async with Store(path) as store:
        rows = await store.db.execute_fetchall(
            "SELECT source_key, count FROM source_counts"
        )

    assert _table(path) == _COLUMNS
    assert _version(path) == 10
    assert [tuple(r) for r in rows] == [("k", 3)]


async def test_migration_9_is_idempotent_over_a_table_that_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store left half-migrated: the table created, the version not moved."""
    path = tmp_path / "s.db"
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", _MIGRATIONS[:8])
        async with Store(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(_MIGRATIONS[8])
        conn.execute(
            "INSERT INTO llm_runs (run_id, ran, postings) VALUES ('r1', 'x', 4)"
        )
        conn.commit()
    assert _version(path) == 8

    async with Store(path):
        pass
    async with Store(path):
        pass

    assert _version(path) == 10
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT run_id, postings FROM llm_runs").fetchall() == [
            ("r1", 4)
        ]


async def test_a_store_newer_than_this_code_is_still_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    async with Store(path):
        pass
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA user_version={len(_MIGRATIONS) + 1}")
    before = path.read_bytes()

    with pytest.raises(StoreTooNewError, match="newer rolescan"):
        async with Store(path):
            pass

    assert path.read_bytes() == before


def _run(**kw: object) -> LlmRun:
    base: dict[str, object] = {"backend": "ollama", "model": "m:1", "postings": 10}
    base.update(kw)
    return LlmRun(**base)  # type: ignore[arg-type]


async def test_a_run_round_trips_through_the_store(tmp_path: Path) -> None:
    run = LlmRun(
        run_id="abc",
        ran="2026-01-02T03:04:05+00:00",
        backend="ollama",
        model="m:1",
        model_digest="0123456789ab",
        calls=7,
        cached=3,
        errors=1,
        breaker=True,
        deferred=2,
        quotes_rejected=4,
        level_overridden=5,
        field_overridden=6,
        years_set=7,
        years_cleared=8,
        bars_added=9,
        postings=10,
    )
    async with Store(tmp_path / "s.db") as store:
        await store.record_llm_run(run)
        [back] = await store.recent_llm_runs()

    assert back == run


async def test_recent_runs_are_newest_first_and_capped(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        for day in range(1, 8):
            await store.record_llm_run(
                _run(ran=f"2026-01-0{day}T00:00:00+00:00", calls=day)
            )

        newest = await store.recent_llm_runs(5)

    assert [r.calls for r in newest] == [7, 6, 5, 4, 3]


async def test_recent_runs_leave_out_a_run_that_finished_no_facts(
    tmp_path: Path,
) -> None:
    """A run with no postings has no rates, so it is no baseline."""
    async with Store(tmp_path / "s.db") as store:
        await store.record_llm_run(_run(ran="2026-01-01T00:00:00+00:00", calls=1))
        await store.record_llm_run(
            _run(ran="2026-01-02T00:00:00+00:00", calls=2, postings=0)
        )

        assert [r.calls for r in await store.recent_llm_runs()] == [1]


async def test_recent_runs_can_be_limited_to_one_model(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.record_llm_run(
            _run(ran="2026-01-01T00:00:00+00:00", model="a:1", calls=1)
        )
        await store.record_llm_run(
            _run(ran="2026-01-02T00:00:00+00:00", model="b:2", calls=2)
        )
        await store.record_llm_run(
            _run(
                ran="2026-01-03T00:00:00+00:00",
                backend="anthropic",
                model="a:1",
                calls=3,
            )
        )

        same = await store.recent_llm_runs(5, backend="ollama", model="a:1")
        every = await store.recent_llm_runs(5)

    assert [r.calls for r in same] == [1]
    assert [r.calls for r in every] == [3, 2, 1]


# --- the counters `finish_facts` keeps ---------------------------------------


def _job(title: str = "Receptionist", description: str = "Answer the phone.") -> Job:
    return Job(
        source="t",
        company="Acme",
        title=title,
        url="https://acme.example/1",
        description=description,
    )


def _raw(**kw: object) -> PostingFacts:
    return PostingFacts.model_validate({"fit_score": 60, "reason": "ok", **kw})


def _counted(raw: PostingFacts, job: Job) -> FactsCounter:
    counter = FactsCounter()
    finish_facts(raw, job, counter)
    return counter


def test_a_fresh_counter_is_all_zero() -> None:
    assert FactsCounter() == FactsCounter(
        postings=0,
        quotes_rejected=0,
        level_overridden=0,
        field_overridden=0,
        years_set=0,
        years_cleared=0,
        bars_added=0,
    )


def test_a_posting_the_model_and_the_code_agree_on_changes_nothing() -> None:
    counter = _counted(_raw(), _job())

    assert counter == FactsCounter(postings=1)


def test_a_quote_that_is_not_in_the_advert_is_counted_as_rejected() -> None:
    raw = _raw(years_required={"value": 5, "quote": "five years of dancing"})

    assert _counted(raw, _job()).quotes_rejected == 1


def test_a_dropped_bar_is_counted_as_rejected() -> None:
    raw = _raw(hard_bars=[{"kind": "clearance", "quote": "never said anywhere"}])

    assert _counted(raw, _job()).quotes_rejected == 1


def test_a_posting_that_lost_two_facts_is_counted_once() -> None:
    raw = _raw(
        years_required={"value": 5, "quote": "five years of dancing"},
        level={"value": "senior", "quote": "a very senior person"},
    )

    assert _counted(raw, _job()).quotes_rejected == 1


def test_a_quote_that_is_in_the_advert_is_not_rejected() -> None:
    job = _job(description="Needs 4 years of experience with SQL.")
    raw = _raw(years_required={"value": 4, "quote": "4 years of experience"})

    assert _counted(raw, job).quotes_rejected == 0


def test_a_title_that_names_a_level_overrides_the_models_level() -> None:
    raw = _raw(level={"value": "mid", "quote": "mid"})

    counter = _counted(raw, _job(title="Senior Data Engineer"))

    assert counter.level_overridden == 1


def test_a_level_the_model_stated_with_a_real_quote_is_not_overridden() -> None:
    job = _job(description="Suited to a mid level engineer with some history.")
    raw = _raw(level={"value": "mid", "quote": "mid level engineer"})

    assert _counted(raw, job).level_overridden == 0


def test_a_field_the_title_names_overrides_the_models_field() -> None:
    raw = _raw(field={"value": "software", "quote": "Answer the phone."})

    counter = _counted(raw, _job(title="Machine Learning Engineer"))

    assert counter.field_overridden == 1


def test_years_the_model_left_out_are_counted_when_the_advert_states_them() -> None:
    job = _job(description="You bring 3+ years of experience in Python.")

    counter = _counted(_raw(), job)

    assert (counter.years_set, counter.years_cleared) == (1, 0)


def test_the_top_of_a_range_corrected_to_its_low_end_counts_as_set() -> None:
    job = _job(description="You have 3-5 years of experience in Python.")
    raw = _raw(years_required={"value": 5, "quote": "3-5 years of experience"})

    counter = _counted(raw, job)

    assert (counter.years_set, counter.years_cleared) == (1, 0)


def test_years_the_advert_does_not_require_are_counted_when_cleared() -> None:
    job = _job(description="You have 0-2 years of experience in SQL.")
    raw = _raw(years_required={"value": 2, "quote": "0-2 years of experience"})

    counter = _counted(raw, job)

    assert (counter.years_set, counter.years_cleared) == (0, 1)


def test_a_bar_the_advert_states_and_the_model_missed_is_counted_as_added() -> None:
    job = _job(title="Data Analyst - UAE National, Analytics Team")

    counter = _counted(_raw(), job)

    assert counter.bars_added == 1


def test_a_bar_the_model_already_named_is_not_counted_as_added() -> None:
    job = _job(title="Data Analyst - UAE National, Analytics Team")
    raw = _raw(hard_bars=[{"kind": "nationality", "quote": "UAE National"}])

    assert _counted(raw, job).bars_added == 0


def test_the_counter_adds_up_over_postings() -> None:
    counter = FactsCounter()
    finish_facts(_raw(), _job(), counter)
    finish_facts(_raw(), _job(title="Senior Data Engineer"), counter)
    finish_facts(
        _raw(years_required={"value": 5, "quote": "five years of dancing"}),
        _job(),
        counter,
    )

    assert counter.postings == 3
    assert counter.level_overridden == 1
    assert counter.quotes_rejected == 1


@pytest.mark.parametrize(
    ("raw", "job"),
    [
        (_raw(), _job()),
        (_raw(), _job(title="Senior Machine Learning Engineer")),
        (
            _raw(years_required={"value": 5, "quote": "five years of dancing"}),
            _job(),
        ),
        (
            _raw(
                level=LevelFact(value=Level.mid, quote="mid"),
                field=FieldFact(value=JobField.software, quote="Answer the phone."),
                years_required=YearsFact(value=5, quote="3-5 years of experience"),
                hard_bars=[HardBar(kind=BarKind.clearance, quote="Answer the phone.")],
            ),
            _job(description="Answer the phone. 3-5 years of experience."),
        ),
        (_raw(), _job(title="Data Analyst - UAE National, Analytics Team")),
    ],
)
def test_counting_never_changes_what_finish_facts_returns(
    raw: PostingFacts, job: Job
) -> None:
    assert finish_facts(raw, job, FactsCounter()) == finish_facts(raw, job)


# --- the scorer's per-run record ---------------------------------------------


class _Canned(Judge):
    """Answers every posting with the same raw facts, or fails when told to."""

    name = "canned-test-judge"

    def __init__(self, raw: PostingFacts, *, fail: bool = False) -> None:
        super().__init__(LLMConfig())
        self.raw = raw
        self.fail = fail
        self.calls = 0

    async def verdict(self, system: str, user: str) -> FitVerdict:
        raise NotImplementedError

    async def facts(self, system: str, user: str) -> PostingFacts:
        self.calls += 1
        if self.fail:
            msg = "model runner crashed"
            raise RuntimeError(msg)
        return self.raw


def _postings(titles: list[str]) -> list[ScoredJob]:
    return [
        ScoredJob(
            job=Job(
                source="t",
                company=f"Acme {i}",
                title=title,
                url=f"https://acme.example/{i}",
                description="Answer the phone.",
            ),
            keyword_score=20,
        )
        for i, title in enumerate(titles)
    ]


def _scorer(
    raw: PostingFacts, store: Store | None = None, *, fail: bool = False, **cfg: object
) -> tuple[FitScorer, _Canned]:
    config = LLMConfig(
        enabled=True, backend="ollama", model="m:1", max_concurrent=1, **cfg
    )
    scorer = FitScorer(config, ProfileConfig(), store, model_digest="0123456789ab")
    judge = _Canned(raw, fail=fail)
    scorer._judge = judge
    return scorer, judge


async def test_a_scorer_that_did_nothing_has_no_record() -> None:
    scorer, _ = _scorer(_raw())

    assert scorer.run_record() is None
    await scorer.score_all([])
    assert scorer.run_record() is None


async def test_the_scorer_records_its_calls_and_what_the_code_changed() -> None:
    scorer, judge = _scorer(_raw())

    await scorer.score_all(
        _postings(["Receptionist", "Senior Machine Learning Engineer", "Receptionist"])
    )

    run = scorer.run_record()
    assert run is not None
    assert (run.backend, run.model, run.model_digest) == (
        "ollama",
        "m:1",
        "0123456789ab",
    )
    assert (run.calls, run.cached, run.errors, run.deferred) == (3, 0, 0, 0)
    assert run.breaker is False
    assert run.postings == 3
    assert run.level_overridden == 1
    assert run.field_overridden == 1
    assert judge.calls == 3


async def test_a_cache_hit_is_counted_and_its_facts_are_finished_again(
    tmp_path: Path,
) -> None:
    """The resolvers run on every read, so a cached posting counts as a posting
    whose facts were finished, and a call it did not make is not a call."""
    titles = ["Senior Machine Learning Engineer", "Receptionist"]
    async with Store(tmp_path / "s.db") as store:
        first, _ = _scorer(_raw(), store)
        await first.score_all(_postings(titles))

        second, judge = _scorer(_raw(), store)
        await second.score_all(_postings(titles))

    run = second.run_record()
    assert run is not None
    assert (run.calls, run.cached, run.postings) == (0, 2, 2)
    assert run.level_overridden == 1
    assert judge.calls == 0


async def test_failed_calls_and_a_tripped_breaker_are_recorded() -> None:
    scorer, _ = _scorer(_raw(), fail=True)

    await scorer.score_all(_postings(["Receptionist"] * 8))

    run = scorer.run_record()
    assert run is not None
    assert run.errors == 5
    assert run.breaker is True
    assert run.deferred == 3
    assert run.postings == 0


async def test_a_judge_mode_run_records_calls_and_no_facts() -> None:
    class _Verdicts(Judge):
        name = "verdict-test-judge"

        async def verdict(self, system: str, user: str) -> FitVerdict:
            return FitVerdict(
                fit_score=50, verdict="consider", confidence="low", reason="ok"
            )

    config = LLMConfig(enabled=True, backend="ollama", mode="judge", cascade=False)
    scorer = FitScorer(config, ProfileConfig())
    scorer._judge = _Verdicts(config)

    await scorer.score_all(_postings(["Receptionist", "Receptionist"]))

    run = scorer.run_record()
    assert run is not None
    assert (run.calls, run.postings, run.level_overridden) == (2, 0, 0)


# --- the health check ---------------------------------------------------------


def _history(n: int, postings: int = 20, **counts: int) -> list[LlmRun]:
    """`n` earlier runs of `postings` postings each, newest first."""
    return [LlmRun(postings=postings, **counts) for _ in range(n)]  # type: ignore[arg-type]


def test_a_rate_over_twice_the_median_of_the_last_runs_is_flagged() -> None:
    this = LlmRun(postings=20, level_overridden=14)

    flags = model_health(this, _history(4, level_overridden=6))

    assert flags == [HealthFlag("level overridden", 14, 20, 0.3, 4)]


def test_a_flag_says_how_many_runs_made_its_median() -> None:
    this = LlmRun(postings=20, level_overridden=14)

    for n in (3, 4, 5):
        assert [
            f.runs for f in model_health(this, _history(n, level_overridden=6))
        ] == [n]
    # Never more than the five that set the baseline, and a run that finished
    # no postings is not one of them.
    padded = [*_history(2, level_overridden=6), LlmRun(postings=0)]
    padded += _history(7, level_overridden=6)
    assert [f.runs for f in model_health(this, padded)] == [5]


def test_a_rate_at_exactly_twice_the_median_is_not_flagged() -> None:
    this = LlmRun(postings=20, level_overridden=12)

    assert model_health(this, _history(4, level_overridden=6)) == []


def test_a_rate_that_is_not_higher_than_before_is_not_flagged() -> None:
    this = LlmRun(postings=20, level_overridden=6, field_overridden=5)

    assert model_health(this, _history(4, level_overridden=6, field_overridden=5)) == []


def test_fewer_than_three_earlier_runs_is_silent() -> None:
    this = LlmRun(postings=20, level_overridden=20)

    assert model_health(this, _history(2, level_overridden=2)) == []
    assert model_health(this, []) == []


def test_three_earlier_runs_are_enough() -> None:
    this = LlmRun(postings=20, level_overridden=20)

    assert [f.label for f in model_health(this, _history(3, level_overridden=2))] == [
        "level overridden"
    ]


def test_an_earlier_run_that_finished_no_facts_is_no_baseline() -> None:
    this = LlmRun(postings=20, level_overridden=20)
    earlier = [*_history(2, level_overridden=2), LlmRun(), LlmRun(calls=9)]

    assert model_health(this, earlier) == []


def test_a_run_of_fewer_than_five_postings_is_silent() -> None:
    this = LlmRun(postings=4, level_overridden=4)

    assert model_health(this, _history(5, level_overridden=2)) == []


def test_a_run_of_exactly_five_postings_can_be_flagged() -> None:
    this = LlmRun(postings=5, level_overridden=5)

    assert [f.label for f in model_health(this, _history(5, level_overridden=2))] == [
        "level overridden"
    ]


def test_only_the_trailing_five_runs_set_the_median() -> None:
    newest = [
        *_history(2, level_overridden=10),
        *_history(3, level_overridden=2),
    ]
    older = _history(3, level_overridden=18)
    this = LlmRun(postings=20, level_overridden=6)

    assert [f.label for f in model_health(this, newest + older)] == ["level overridden"]
    assert model_health(this, older + newest) == []


def test_one_odd_run_among_the_last_five_does_not_lift_the_bar() -> None:
    """A median, not a mean: the mean of these five is 0.44, the median 0.3."""
    earlier = [*_history(4, level_overridden=6), *_history(1, level_overridden=20)]
    this = LlmRun(postings=20, level_overridden=14)

    assert [f.label for f in model_health(this, earlier)] == ["level overridden"]


def test_against_a_median_of_zero_one_posting_is_not_enough() -> None:
    earlier = _history(4)

    assert model_health(LlmRun(postings=20, bars_added=1), earlier) == []
    assert model_health(LlmRun(postings=20, bars_added=4), earlier) == []
    assert [
        f.label for f in model_health(LlmRun(postings=20, bars_added=5), earlier)
    ] == ["bars added"]


def test_each_rate_is_judged_on_its_own() -> None:
    earlier = _history(
        4,
        quotes_rejected=2,
        level_overridden=6,
        field_overridden=5,
        years_set=2,
        years_cleared=1,
        bars_added=1,
    )
    this = LlmRun(
        postings=20,
        quotes_rejected=2,
        level_overridden=6,
        field_overridden=5,
        years_set=12,
        years_cleared=1,
        bars_added=1,
    )

    assert model_health(this, earlier) == [
        HealthFlag("years filled in", 12, 20, 0.1, 4)
    ]


def test_every_rate_has_a_flag_when_every_rate_jumps() -> None:
    this = LlmRun(
        postings=20,
        quotes_rejected=20,
        level_overridden=20,
        field_overridden=20,
        years_set=20,
        years_cleared=20,
        bars_added=20,
    )

    labels = [f.label for f in model_health(this, _history(5, level_overridden=1))]

    assert labels == [
        "quotes rejected",
        "level overridden",
        "field overridden",
        "years filled in",
        "years cleared",
        "bars added",
    ]
