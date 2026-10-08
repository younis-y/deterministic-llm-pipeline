"""The facts cache holds what the model said, keyed on what it was asked
(2.5.8).

Two defects from 2026-10-07. The key was a hand-bumped `facts-vN` plus the
examples digest: editing the prompt or the candidate summary replayed facts
made for the old one for `cache_days`, and nothing named the model's weights.
And the cache held facts AFTER the resolvers, so every resolver fix needed a
bump, and the 200-posting evaluation re-ran the model (36-46 minutes) to
test a regular expression. Now the row holds the raw facts, `finish_facts`
runs on every read, and the key ends in a fingerprint of everything that
decides the answer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

import rolescan.scoring.llm as llm
from rolescan.config import Config, LLMConfig, ProfileConfig
from rolescan.digest import render_markdown
from rolescan.models import FitVerdict, Job, ScoredJob
from rolescan.pipeline import ScanResult, _settle_identity, run_scan
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import PostingFacts, YearsFact
from rolescan.scoring.judges import Judge, served_model_digest
from rolescan.scoring.llm import SYSTEM_FACTS, cache_key, prompt_fingerprint
from rolescan.store import Store

RAW = PostingFacts.model_validate({"fit_score": 70, "reason": "Strong SQL overlap."})


class _Counting(Judge):
    """Returns `RAW` and counts the calls."""

    name = "counting-test-judge"

    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self.calls = 0

    async def verdict(self, system: str, user: str) -> FitVerdict:
        raise NotImplementedError

    async def facts(self, system: str, user: str) -> PostingFacts:
        self.calls += 1
        return RAW


def _posting() -> ScoredJob:
    return ScoredJob(
        job=Job(
            source="t",
            company="Acme",
            title="Graduate Data Engineer",
            url="https://acme.example/1",
            description="Build pipelines. 3+ years of experience in SQL.",
        ),
        keyword_score=20,
    )


def _scorer(
    store: Store | None = None, *, summary: str = "A candidate.", **cfg: object
) -> tuple[FitScorer, _Counting]:
    config = LLMConfig(enabled=True, backend="ollama", mode="facts", **cfg)
    scorer = FitScorer(config, ProfileConfig(summary=summary), store)
    judge = _Counting()
    scorer._judge = judge
    return scorer, judge


def test_the_prompt_fingerprint_is_pinned() -> None:
    """Fails when the facts prompt, the user template, the schema or a
    default setting changes. That is deliberate: such a change re-asks every
    cached posting and moves the evaluation's numbers, so update this value
    in the same commit, and re-run the evaluation."""
    fingerprint = prompt_fingerprint(
        SYSTEM_FACTS.format(summary="A candidate."), LLMConfig()
    )
    assert fingerprint == "439fe8318511"


@pytest.mark.parametrize(
    "change",
    [
        {"summary": "A finance candidate."},
        {"num_ctx": 16384},
        {"description_chars": 4000},
        {"temperature": 0.7},
        {"model": "another:1"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_input_to_the_answer_moves_the_key(change: dict[str, object]) -> None:
    job = _posting().job
    base, _ = _scorer()
    other, _ = _scorer(**change)  # type: ignore[arg-type]
    assert base._facts_cache_key(job) != other._facts_cache_key(job)


def test_the_model_digest_and_the_prompt_text_move_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _posting().job
    base, _ = _scorer()
    repulled = FitScorer(base.cfg, base.profile, model_digest="0123456789ab")
    assert base._facts_cache_key(job) != repulled._facts_cache_key(job)

    monkeypatch.setattr(llm, "SYSTEM_FACTS", SYSTEM_FACTS + "\nOne more rule.")
    edited, _ = _scorer()
    assert base._facts_cache_key(job) != edited._facts_cache_key(job)


async def test_the_cache_holds_raw_facts_and_a_resolver_change_reaches_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posting = _posting()
    async with Store(tmp_path / "s.db") as store:
        first, judge = _scorer(store)
        await first.score_all([posting])
        raw = await store.get_verdict(
            first._facts_cache_key(posting.job), 30, PostingFacts
        )
        assert judge.calls == 1
        assert raw == RAW, "what the model said, before the resolvers"
        assert first.last_facts[posting.job.url].years_required.value == 3

        def stricter(facts: PostingFacts, job: Job) -> PostingFacts:
            return facts.model_copy(
                update={"years_required": YearsFact(value=5, quote="3+ years")}
            )

        monkeypatch.setattr(llm, "resolve_years", stricter)
        second, judge = _scorer(store)
        [out] = await second.score_all([posting])

    assert judge.calls == 0, "a resolver change costs no model call"
    assert out.llm_cached
    assert second.last_facts[posting.job.url].years_required.value == 5


async def test_a_row_cached_by_2_5_7_is_never_read(tmp_path: Path) -> None:
    """A v13 row holds facts that were already resolved once."""
    posting = _posting()
    async with Store(tmp_path / "s.db") as store:
        scorer, judge = _scorer(store)
        old_key = f"{posting.job.content_hash}:facts-v13:ollama:{scorer.cfg.model}"
        await store.put_verdict(old_key, RAW.model_copy(update={"fit_score": 99}))

        [out] = await scorer.score_all([posting])

    assert judge.calls == 1
    assert out.fit is not None and out.fit.fit_score == 70


def test_the_examples_digest_keyword_still_keys_a_call_with_no_fingerprint() -> None:
    """`cache_key` keeps every keyword it ever had. With no fingerprint the
    examples digest suffixes the key as it did before 2.5.8; given a
    fingerprint, which already covers the examples, the suffix is not added."""
    job = _posting().job
    cfg = LLMConfig(enabled=True, backend="ollama", model="m:1")
    base = f"{job.content_hash}:facts-v14:ollama:m:1"

    assert cache_key(job, "facts", cfg, examples_digest="abc123") == f"{base}:ex-abc123"
    assert (
        cache_key(job, "facts", cfg, fingerprint="0123456789ab", examples_digest="x")
        == f"{base}:0123456789ab"
    )


_FACTS_JSON = json.dumps(RAW.model_dump(mode="json"))
_BOARD = {
    "jobs": [
        {
            "id": 1,
            "title": "Energy Data Analyst",
            "location": {"name": "London, UK"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "content": "<p>Python and energy markets.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        }
    ]
}


def _tags(digest: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": "m:1", "digest": digest}]})


@respx.mock
async def test_a_re_pulled_model_is_asked_again(tmp_path: Path) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=[_tags("a" * 64), _tags("b" * 64), _tags("b" * 64)]
    )
    respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, json={"model_info": {}})
    )
    chat = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": _FACTS_JSON}})
    )
    cfg = Config.model_validate(
        {
            "profile": {"keywords": {"energy": 6}, "min_keyword_score": 1},
            "llm": {"enabled": True, "backend": "ollama", "model": "m:1"},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "s.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )

    await run_scan(cfg, dry_run=True)
    await run_scan(cfg, dry_run=True)
    third = await run_scan(cfg, dry_run=True)

    assert chat.call_count == 2, "new weights miss the cache; the same weights hit"
    assert third.llm_cached == 1


# --- a probe that could not name the weights must not touch the cache --------


def _snapshot(db: Path) -> dict[str, str]:
    """Every verdict row: key -> payload."""
    with sqlite3.connect(db) as conn:
        return dict(conn.execute("SELECT content_hash, payload FROM verdicts"))


def _ollama_cfg(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "profile": {"keywords": {"energy": 6}, "min_keyword_score": 1},
            "llm": {"enabled": True, "backend": "ollama", "model": "m:1"},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "s.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


def _mock_board_and_chat() -> respx.Route:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, json={"model_info": {}})
    )
    return respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": _FACTS_JSON}})
    )


@respx.mock
async def test_a_probe_that_times_out_neither_reads_nor_writes_the_facts_cache(
    tmp_path: Path,
) -> None:
    """healthy / tags times out (twice: the probe and its longer re-read) /
    healthy. The middle run cannot say which weights it is talking to, so it
    must not be served rows keyed on weights it cannot confirm, and must not
    write a row under a key that names no weights (a later run after a
    re-pull would be handed facts the old weights made)."""
    chat = _mock_board_and_chat()
    timeout = httpx.ReadTimeout("timed out")
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=[_tags("a" * 64), timeout, timeout, _tags("a" * 64)]
    )
    cfg = _ollama_cfg(tmp_path)

    first = await run_scan(cfg, dry_run=True)
    after_first = _snapshot(tmp_path / "s.db")
    assert chat.call_count == 1 and not first.facts_cache_skipped

    second = await run_scan(cfg, dry_run=True)
    assert chat.call_count == 2, "the cache was not read: the model was asked again"
    assert second.llm_cached == 0
    assert second.facts_cache_skipped
    assert _snapshot(tmp_path / "s.db") == after_first, "and nothing was written"
    assert "facts cache skipped: model identity unknown" in render_markdown(second)

    third = await run_scan(cfg, dry_run=True)
    assert chat.call_count == 2, "the first run's row is read again"
    assert third.llm_cached == 1 and not third.facts_cache_skipped


@respx.mock
async def test_a_slow_probe_is_re_read_once_and_then_trusted(tmp_path: Path) -> None:
    """The 3 s probe misses, the 10 s re-read answers: the digest it reads is
    the run's identity, so the cache works as on a healthy run."""
    chat = _mock_board_and_chat()
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=[
            httpx.ReadTimeout("timed out"),
            _tags("a" * 64),
            _tags("a" * 64),
        ]
    )
    cfg = _ollama_cfg(tmp_path)

    slow = await run_scan(cfg, dry_run=True)
    again = await run_scan(cfg, dry_run=True)

    assert not slow.facts_cache_skipped
    assert slow.llm_model_digest == "a" * 12
    assert chat.call_count == 1 and again.llm_cached == 1


@respx.mock
async def test_a_server_that_reports_no_digest_keeps_its_cache(tmp_path: Path) -> None:
    """An Ollama-compatible server whose model list carries no digest has no
    identity to read: "" is the honest answer, not an unknown one."""
    chat = _mock_board_and_chat()
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m:1"}]})
    )
    cfg = _ollama_cfg(tmp_path)

    first = await run_scan(cfg, dry_run=True)
    second = await run_scan(cfg, dry_run=True)

    assert not first.facts_cache_skipped and not second.facts_cache_skipped
    assert chat.call_count == 1 and second.llm_cached == 1


@respx.mock
async def test_a_healthy_probe_with_no_digest_field_is_not_read_again(
    tmp_path: Path,
) -> None:
    """The longer re-read of the model list is for a probe that reported a
    problem. A probe that answered, listing the model with no digest, is
    healthy: re-reading would only ask the same server the same question, so
    each run makes exactly one `/api/tags` request."""
    _mock_board_and_chat()
    tags = respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m:1"}]})
    )
    cfg = _ollama_cfg(tmp_path)

    result = await run_scan(cfg, dry_run=True)

    assert not result.llm_unusable and result.llm_model_digest == ""
    assert tags.call_count == 1

    await run_scan(cfg, dry_run=True)
    assert tags.call_count == 2, "one probe per run, no second read"


@respx.mock
async def test_a_backend_that_names_no_digest_is_unaffected(tmp_path: Path) -> None:
    """A hosted backend has no digest to read and makes no request to look
    for one (any unmocked request would fail this test)."""
    cfg = Config.model_validate(
        {
            "llm": {
                "enabled": True,
                "backend": "anthropic",
                "api_key": "k",
                "model": "claude-x",
            },
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "s.db")},
        }
    )
    result = ScanResult()

    await _settle_identity(cfg, result)

    assert result.llm_model_digest == "" and not result.facts_cache_skipped
    assert await served_model_digest(cfg.llm) == ""


@respx.mock
async def test_a_model_the_server_does_not_list_has_an_unknown_identity() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=_tags("a" * 64)  # lists "m:1"
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="other:2")

    assert await served_model_digest(cfg) is None
    assert await served_model_digest(cfg.model_copy(update={"model": "m:1"})) == (
        "a" * 12
    )


async def test_a_scorer_with_the_facts_cache_off_neither_reads_nor_writes(
    tmp_path: Path,
) -> None:
    posting = _posting()
    async with Store(tmp_path / "s.db") as store:
        warm, _ = _scorer(store)
        await warm.score_all([posting])
        before = await _rows(store)

        cold = FitScorer(warm.cfg, warm.profile, store, facts_cache=False)
        judge = _Counting()
        cold._judge = judge
        [out] = await cold.score_all([posting])
        after = await _rows(store)

    assert judge.calls == 1 and not out.llm_cached
    assert out.fit is not None and out.fit.fit_score == 70
    assert cold.last_facts[posting.job.url].years_required.value == 3
    assert after == before, "no facts row was written"


async def _rows(store: Store) -> set[str]:
    cur = await store.db.execute("SELECT content_hash FROM verdicts")
    return {row[0] for row in await cur.fetchall()}
