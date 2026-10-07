from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from conftest import OLLAMA_MODEL, mock_ollama
from rolescan.config import Config
from rolescan.digest import render_html, render_markdown
from rolescan.models import (
    Confidence,
    FitVerdict,
    Job,
    ScoredJob,
    Verdict,
)
from rolescan.pipeline import (
    ScanResult,
    SourceReport,
    _check_coverage,
    _drop_stale,
    _preflight,
    _rule_hidden,
    deduplicate,
    run_scan,
)
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import PostingFacts
from rolescan.store import Store


def _payload(title: str, content: str, jid: int = 1) -> dict[str, object]:
    return {
        "jobs": [
            {
                "id": jid,
                "title": title,
                "location": {"name": "London, UK"},
                "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                "content": content,
                "updated_at": "2026-08-20T10:00:00Z",
            }
        ]
    }


def test_deduplicate_keeps_the_richest_copy() -> None:
    thin = Job(
        source="adzuna",
        company="Drax",
        title="Energy Analyst",
        location="London",
        url="https://adzuna/1",
        description="short",
    )
    rich = Job(
        source="greenhouse",
        company="Drax",
        title="Energy Analyst",
        location="London",
        url="https://drax/1",
        description="a much longer description with real detail",
    )
    out = deduplicate([thin, rich])
    assert len(out) == 1
    assert out[0].source == "greenhouse", "prefer the ATS copy over the aggregator"


def test_deduplicate_is_order_independent() -> None:
    a = Job(
        source="x",
        company="C",
        title="T",
        location="L",
        url="https://1",
        description="aa",
    )
    b = Job(
        source="y",
        company="C",
        title="T",
        location="L",
        url="https://2",
        description="aaaa",
    )
    assert deduplicate([a, b])[0].description == deduplicate([b, a])[0].description


@respx.mock
async def test_scan_reports_a_match(config: Config) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    result = await run_scan(config)
    assert result.unique == 1
    assert len(result.reportable) == 1
    assert result.reportable[0].job.company == "Acme"


@respx.mock
async def test_second_run_reports_nothing_new(config: Config) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    assert len((await run_scan(config)).reportable) == 1
    second = await run_scan(config)
    assert second.reportable == []
    assert second.already_seen == 1


@respx.mock
async def test_dry_run_does_not_persist(config: Config) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload("Graduate Energy Data Scientist", "Python, trading, energy."),
        )
    )
    await run_scan(config, dry_run=True)
    again = await run_scan(config, dry_run=True)
    assert again.already_seen == 0, "a dry run must leave the store untouched"


@respx.mock
async def test_gated_role_is_filtered_by_prefilter(config: Config) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Data Scientist",
                "Analytics role. UAE National (National Talent programme) required.",
            ),
        )
    )
    result = await run_scan(config)
    assert result.reportable == []
    assert result.prefiltered == 1


@respx.mock
async def test_a_dead_source_does_not_kill_the_scan(config: Config) -> None:
    config.sources.append(
        type(config.sources[0])(kind="greenhouse", slug="dead", label="Dead")
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload("Graduate Energy Data Scientist", "Python, trading, energy."),
        )
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/dead/jobs").mock(
        return_value=httpx.Response(404)
    )
    result = await run_scan(config)
    assert len(result.reportable) == 1
    assert len(result.failed_sources) == 1
    assert "404" in result.failed_sources[0].error


# --- digest ---------------------------------------------------------------


def _scored(job: Job, **kw: object) -> ScoredJob:
    base = {
        "fit_score": 82,
        "verdict": Verdict.APPLY,
        "confidence": Confidence.HIGH,
        "reason": "Overlaps the day-ahead forecasting work.",
    }
    return ScoredJob(job=job, fit=FitVerdict.model_validate(base | kw))


def test_digest_renders_llm_detail(energy_job: Job) -> None:
    result = ScanResult(
        unique=5,
        reportable=[_scored(energy_job)],
        reports=[SourceReport("greenhouse", "acme", "Acme", 5)],
    )
    out = render_markdown(result)
    assert "EDF Trading" in out
    assert "82/100" in out
    assert "[Apply](https://example.com/1)" in out


def test_digest_separates_blocked(energy_job: Job, gated_job: Job) -> None:
    result = ScanResult(
        unique=2,
        reportable=[
            _scored(energy_job),
            _scored(
                gated_job,
                verdict=Verdict.BLOCKED,
                fit_score=70,
                blockers=["UAE National only"],
            ),
        ],
    )
    out = render_markdown(result)
    assert "## Worth a look" in out
    assert "## Blocked" in out
    assert out.index("## Worth a look") < out.index("## Blocked")
    assert "UAE National only" in out


def test_empty_digest_says_so_plainly() -> None:
    out = render_markdown(ScanResult(unique=40, already_seen=40))
    assert "Nothing new worth your time" in out
    assert "40" in out


def test_digest_surfaces_failed_sources() -> None:
    result = ScanResult(
        reports=[
            SourceReport("greenhouse", "dead", "Dead", error="FetchError: HTTP 404")
        ]
    )
    out = render_markdown(result)
    assert "greenhouse/dead" in out
    assert "rolescan discover" in out


def test_dry_run_is_labelled_in_the_digest(energy_job: Job) -> None:
    out = render_markdown(ScanResult(reportable=[_scored(energy_job)], dry_run=True))
    assert "Dry run" in out


def test_blocked_roles_get_no_cv_advice(gated_job: Job) -> None:
    """Suggesting a CV for a role you cannot be considered for invites waste."""
    item = _scored(
        gated_job,
        verdict=Verdict.BLOCKED,
        fit_score=70,
        blockers=["UAE National only"],
    )
    out = render_markdown(ScanResult(unique=1, reportable=[item]))
    assert "UAE National only" in out
    assert "**Send:**" not in out


def test_stats_line_pluralises() -> None:
    one = render_markdown(
        ScanResult(unique=1, reports=[SourceReport("greenhouse", "a", "A", 1)])
    )
    assert "1 unique posting from 1 source." in one
    many = render_markdown(
        ScanResult(
            unique=9,
            reports=[
                SourceReport("greenhouse", "a", "A", 1),
                SourceReport("lever", "b", "B", 8),
            ],
        )
    )
    assert "9 unique postings from 2 sources." in many


# --- the structured source's posting cache must survive across scans -------


@respx.mock
async def test_a_second_scan_does_not_refetch_an_unchanged_posting(
    tmp_path: Path,
) -> None:
    """The whole point of the lastmod cache: 347 pages a day, almost all
    unchanged. If run_scan does not hand the source a cache, it re-downloads
    everything and the cache is decorative."""
    sitemap = (
        '<?xml version="1.0"?><urlset><url>'
        "<loc>https://ex.test/job/1</loc><lastmod>2026-08-25</lastmod>"
        "</url></urlset>"
    )
    page = (
        '<html><head><script type="application/ld+json">'
        '{"@context":"http://schema.org","@type":"JobPosting",'
        '"title":"Power Market Analyst","datePosted":"2026-08-23",'
        '"hiringOrganization":{"@type":"Organization","name":"Ex Energy"},'
        '"description":"&lt;p&gt;Forecasting day-ahead electricity prices.&lt;/p&gt;"}'
        "</script></head><body>x</body></html>"
    )
    respx.get("https://ex.test/sitemap.xml").mock(
        return_value=httpx.Response(200, text=sitemap)
    )
    detail = respx.get("https://ex.test/job/1").mock(
        return_value=httpx.Response(200, text=page)
    )

    cfg = Config.model_validate(
        {
            "profile": {"min_keyword_score": 0, "min_report_score": 0},
            "llm": {"enabled": False},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [
                {
                    "kind": "structured",
                    "slug": "ex",
                    "label": "Ex Energy",
                    "sitemap": "https://ex.test/sitemap.xml",
                    "url_pattern": "/job/",
                    "delay": 0,
                }
            ],
        }
    )

    first = await run_scan(cfg)
    assert first.fetched == 1
    assert detail.call_count == 1

    second = await run_scan(cfg)
    assert second.fetched == 1, "the posting must still reach the pipeline"
    assert detail.call_count == 1, "an unchanged lastmod must not re-download it"
    assert second.already_seen == 1, "and the store must recognise it as seen"


# --- a failing LLM must not look like a successful keyword-only scan -------


@respx.mock
async def test_llm_failure_is_surfaced_not_silently_downgraded(tmp_path: Path) -> None:
    """A revoked ANTHROPIC_API_KEY makes every scoring call 401. Each failure
    was caught per-job and logged at WARNING, so the digest rendered a normal
    scan with every role marked "keyword only" — indistinguishable from a
    deliberate --no-llm run. Twelve hours of postings scored by keywords alone,
    with nothing in the output saying so."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200, json=_payload("Power Market Analyst", "Forecasting day-ahead prices.")
        )
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            401,
            json={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "API key is invalid.",
                },
            },
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {"min_keyword_score": 0, "min_report_score": 0},
            "llm": {"enabled": True, "api_key": "test-key-rejected-by-the-api"},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )
    result = await run_scan(cfg)

    assert result.llm_errors == 1, "the scan must count scoring failures"
    text = render_markdown(result)
    assert "scoring failed" in text.casefold(), text


# --- what the scorer could not judge must not be marked seen ---------------


async def _seen_uids(db_path: Path) -> set[str]:
    from rolescan.store import Store

    async with Store(db_path) as store:
        rows = await store.db.execute_fetchall("SELECT uid FROM seen")
    return {str(r[0]) for r in rows}


#: The same keyword table the `config` fixture uses, so a posting that scores
#: above the prefilter there scores above it here too.
_KEYWORDS = {"energy": 6, "data scientist": 7, "python": 4, "trading": 6}


def _cfg(tmp_path: Path, llm: dict[str, object]) -> Config:
    return Config.model_validate(
        {
            "profile": {
                "keywords": _KEYWORDS,
                "min_keyword_score": 18,
                "min_report_score": 55,
            },
            "llm": llm,
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


@respx.mock
async def test_an_unjudged_posting_is_not_buried_in_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forty-postings defect, in one test.

    `seen` used to be seeded from every fresh posting regardless of whether
    anything had judged it. The configured backend here cannot start at all -
    `anthropic` with no key, which is how the real incident began - so a run
    recorded forty real roles it had formed no opinion of, and `filter_new`
    then suppressed every one of them for good. Nothing in the digest said so,
    because from the outside the run had succeeded.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    cfg = _cfg(tmp_path, {"enabled": True, "backend": "anthropic", "api_key": ""})
    first = await run_scan(cfg)
    assert first.llm_unusable, "the backend was supposed to work and did not"
    assert first.prefiltered == 0, "this posting reached the scorer"
    assert await _seen_uids(tmp_path / "seen.db") == set()

    second = await run_scan(cfg)
    assert second.already_seen == 0, "it must still be reachable tomorrow"


@respx.mock
async def test_a_deliberate_keyword_only_run_does_record_what_it_judged(
    tmp_path: Path,
) -> None:
    """The other side of the line, and the reason the gate is not `enabled`.

    `llm.enabled: false` is not a broken backend. Those postings WERE judged,
    on keywords, which is this user's chosen judgement — so they are recorded
    and not reported again. Refusing to record them would repeat the same
    roles in every digest forever: the same silent failure the rest of this
    wave exists to end, just a quieter one.
    """
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    cfg = _cfg(tmp_path, {"enabled": False})
    first = await run_scan(cfg)
    assert first.llm_unusable == "", "nobody asked for a judge, so none is broken"
    assert first.prefiltered == 0, "this posting reached the scorer"
    assert len(await _seen_uids(tmp_path / "seen.db")) == 1

    second = await run_scan(cfg)
    assert second.already_seen == 1, "a keyword-only run must not repeat itself"


async def test_preflight_reports_an_unknown_enricher_before_fetching(
    tmp_path: Path,
) -> None:
    """A working backend does not hide a misspelled `llm.enricher`: the same
    "fail fast, before anything is fetched" contract a bad backend gets.
    `anthropic` with a (fake) key needs no network for its own preflight, so
    this isolates the enricher check.

    The two reasons are returned separately (round 2 of this task's review):
    an unknown enricher must never be folded into the backend reason, since
    every caller of `llm_unusable` - the digest, the CLI, `_record_assessed`
    - treats a non-empty value as "the judge backend cannot run"."""
    cfg = _cfg(
        tmp_path,
        {
            "enabled": True,
            "backend": "anthropic",
            "api_key": "fake-key",
            "enricher": "not-a-real-enricher",
        },
    )
    backend_reason, enricher_reason = await _preflight(cfg)
    assert backend_reason == "", "the backend itself is fine"
    assert "not-a-real-enricher" in enricher_reason


@respx.mock
async def test_a_posting_the_scorer_errored_on_is_not_buried_either(
    tmp_path: Path,
) -> None:
    """Scoring that was configured, attempted, and failed is still scoring
    that produced no verdict. Recording on the strength of it buries the role
    exactly as thoroughly as recording on no attempt at all."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(500, text="model exploded")
    )
    # The tag this config asks for, so preflight passes and the backend is
    # genuinely usable. A mismatched tag here made preflight report the model
    # as not pulled, and the test then passed from the wrong side of
    # `backend_broke`: on llm_unusable, never on the scoring error it names.
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": OLLAMA_MODEL}]})
    )
    cfg = _cfg(tmp_path, {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL})
    result = await run_scan(cfg)
    assert result.llm_unusable == "", "the backend was usable; the call is what failed"
    assert result.llm_errors == 1
    assert await _seen_uids(tmp_path / "seen.db") == set()


@respx.mock
async def test_a_healthy_run_still_records_what_it_judged(tmp_path: Path) -> None:
    """The other half of the contract: when the scorer answers, the answer is
    recorded, and the role is not repeated tomorrow."""
    mock_ollama()
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    cfg = _cfg(
        tmp_path,
        {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL, "mode": "judge"},
    )
    result = await run_scan(cfg)
    assert len(result.reportable) == 1
    assert len(await _seen_uids(tmp_path / "seen.db")) == 1
    assert (await run_scan(cfg)).already_seen == 1


@respx.mock
async def test_a_healthy_run_with_an_unknown_enricher_scores_normally(
    tmp_path: Path,
) -> None:
    """Round 2 of this task's review: a misspelt `llm.enricher` must not be
    mistaken for a broken judge backend anywhere downstream. The backend here
    is genuinely healthy (`mock_ollama`), so `llm_unusable` must stay empty,
    `enricher_unusable` must carry the (separate) reason, and every normal
    consequence of a healthy run - postings scored, recorded in `seen`, the
    digest's ordinary content - must be unaffected."""
    mock_ollama()
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    cfg = _cfg(
        tmp_path,
        {
            "enabled": True,
            "backend": "ollama",
            "model": OLLAMA_MODEL,
            "mode": "judge",
            "enricher": "not-a-real-enricher",
        },
    )
    result = await run_scan(cfg)

    assert result.llm_unusable == "", "the backend is healthy"
    assert "not-a-real-enricher" in result.enricher_unusable
    assert len(result.reportable) == 1
    assert len(await _seen_uids(tmp_path / "seen.db")) == 1

    md = render_markdown(result)
    assert "not-a-real-enricher" in md
    assert "did not run at all" not in md
    assert "pre-scan backend check failed" not in md

    html = render_html(result)
    assert "not-a-real-enricher" in html
    assert "did not run at all" not in html
    assert "pre-scan backend check failed" not in html


@respx.mock
async def test_a_hard_blocker_overrides_a_high_llm_score(tmp_path: Path) -> None:
    """A term in `hard_blockers` is an instruction, not a hint. A posting that
    clears the keyword prefilter and that the LLM scores well above
    min_report_score must still be excluded once it hits one - the LLM's
    opinion does not get to outvote a structural bar the user configured."""
    mock_ollama()  # canned verdict: fit_score=72, verdict="consider" - not blocked
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting. "
                "10+ years experience required.",
            ),
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {
                # Heavier than _KEYWORDS so the posting still clears
                # min_keyword_score after the 60-point blocker penalty -
                # this is what "reaches the LLM anyway" requires in practice.
                "keywords": {
                    "energy": 20,
                    "data scientist": 20,
                    "python": 15,
                    "trading": 15,
                },
                "blockers": {"10+ years": 60},
                "hard_blockers": ["10+ years"],
                "min_keyword_score": 18,
                "min_report_score": 55,
            },
            "llm": {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL},
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                "show_blocked": False,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )
    result = await run_scan(cfg)
    assert result.prefiltered == 0, "the posting must have reached the LLM"
    assert result.reportable == [], "a hard blocker must not reach the digest"


@respx.mock
async def test_a_blocked_and_hidden_posting_is_counted(tmp_path: Path) -> None:
    """The role is deleted, permanently: dropped from the digest AND written
    to `seen`, so it can never resurface. Nothing counted that. Without a
    number the reader cannot tell a blocker matching the wrong thing from a
    market with nothing in it - and the blocker fires on a substring in a
    config file, over the model's own verdict."""
    mock_ollama()  # canned verdict: fit_score=72, verdict="consider" - not blocked
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting. "
                "Active security clearance required.",
            ),
        )
    )

    def _config(*, show_blocked: bool) -> Config:
        return Config.model_validate(
            {
                "profile": {
                    "keywords": {
                        "energy": 20,
                        "data scientist": 20,
                        "python": 15,
                        "trading": 15,
                    },
                    "hard_blockers": ["security clearance"],
                    "min_keyword_score": 18,
                    "min_report_score": 55,
                },
                "llm": {
                    "enabled": True,
                    "backend": "ollama",
                    "model": OLLAMA_MODEL,
                },
                "output": {
                    "dir": str(tmp_path),
                    "db_path": str(tmp_path / "seen.db"),
                    "show_blocked": show_blocked,
                },
                "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
            }
        )

    result = await run_scan(_config(show_blocked=False), dry_run=True)
    assert result.reportable == [], "the blocked posting is gone from the digest"
    assert result.hidden_blocked == 1, "and something has to say it was here"
    assert "1 blocked and hidden" in render_markdown(result)

    shown = await run_scan(_config(show_blocked=True), dry_run=True)
    assert len(shown.reportable) == 1
    assert shown.hidden_blocked == 0, "nothing was hidden, so nothing to report"
    assert "blocked and hidden" not in render_markdown(shown)


@respx.mock
async def test_a_weighted_term_alone_does_not_override_the_llm(
    tmp_path: Path,
) -> None:
    """A term in `blockers` but not in `hard_blockers` is a preference, not a
    structural bar (matlab: 15 in the real config - "capable, but does not
    want to use it again"). It must cost its weight in keyword_score but must
    not force `blocked`: the LLM's own verdict stands."""
    mock_ollama()  # canned verdict: fit_score=72, verdict="consider"
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting. Some MATLAB use.",
            ),
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": _KEYWORDS,
                "blockers": {"matlab": 15},
                "min_keyword_score": 18,
                "min_report_score": 55,
            },
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "model": OLLAMA_MODEL,
                "mode": "judge",
            },
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                "show_blocked": False,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )
    result = await run_scan(cfg)
    assert result.prefiltered == 0, "the posting must have reached the LLM"
    assert len(result.reportable) == 1, "a soft blocker must not bury the posting"
    assert result.reportable[0].verdict is Verdict.CONSIDER, "the LLM verdict stands"


@respx.mock
async def test_a_prefiltered_reject_is_recorded_even_with_scoring_off(
    tmp_path: Path,
) -> None:
    """Keyword rejects were assessed and rejected on their merits, so they are
    safe to bury. Refusing to record them would re-report every warehouse job
    on every run forever."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200, json=_payload("Warehouse Operative", "Lifting boxes.")
        )
    )
    cfg = _cfg(tmp_path, {"enabled": False})
    result = await run_scan(cfg)
    assert result.prefiltered == 1
    assert len(await _seen_uids(tmp_path / "seen.db")) == 1
    assert (await run_scan(cfg)).already_seen == 1


# --- a dismissed posting must never come back ------------------------------


@respx.mock
async def test_a_dismissed_url_does_not_reappear(tmp_path: Path) -> None:
    """The spec requires it and nothing called `dismissed_urls()`.

    `seen` cannot do this on its own: the same role arrives under a fresh uid
    from the next board that lists it, which is the common case rather than
    the exotic one.
    """
    from rolescan.store import Store

    mock_ollama()
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                "Graduate Energy Data Scientist",
                "Python, trading, energy, day-ahead forecasting.",
            ),
        )
    )
    cfg = _cfg(
        tmp_path,
        {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL, "mode": "judge"},
    )
    first = await run_scan(cfg, dry_run=True)
    assert len(first.reportable) == 1
    url = first.reportable[0].job.url

    async with Store(tmp_path / "seen.db") as store:
        await store.mark(url, "dismissed")

    again = await run_scan(cfg, dry_run=True)
    assert again.reportable == [], "a dismissed posting must never return"
    # Counted with the already-seen rather than vanishing, so the stats line
    # still adds up: unique = already seen + prefiltered + what was scored.
    assert again.unique == 1
    assert again.already_seen == 1
    assert again.prefiltered == 0


# --- an unusable judge must reach the reader, not just the log -------------


@respx.mock
async def test_an_unusable_backend_is_named_in_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest written by a scan that never scored anything reads exactly
    like a quiet day. It is not one, and the difference has to be in the
    email, because the log is the thing nobody opens."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200, json=_payload("Power Market Analyst", "Forecasting day-ahead prices.")
        )
    )
    cfg = _cfg(tmp_path, {"enabled": True, "backend": "anthropic", "api_key": ""})
    result = await run_scan(cfg)
    assert result.llm_unusable, "the scan must notice its judge cannot run"
    assert result.llm_errors == 0, "nothing was attempted, so nothing errored"

    text = render_markdown(result)
    assert "LLM scoring did not run at all" in text
    assert "ANTHROPIC_API_KEY" in text


async def test_no_llm_suppresses_the_unusable_backend_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-llm` is the reader saying they already know."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = Config.model_validate(
        {
            "llm": {"enabled": True, "backend": "anthropic", "api_key": ""},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [],
        }
    )
    result = await run_scan(cfg, check_llm=False)
    assert result.llm_unusable == ""


# --- stale postings and coverage alarms ----------------------------------


def _job_posted(days_ago: int | None, title: str = "Data Engineer") -> Job:
    from datetime import UTC, datetime, timedelta

    return Job(
        # An aggregator, deliberately: greenhouse and the other ATS readers are
        # exempt from ageing, so using one here would make these tests pass for
        # the wrong reason.
        source="linkedin",
        company="Acme",
        title=title,
        location="London",
        url=f"https://example.com/{title}-{days_ago}",
        description="Build pipelines.",
        posted=None
        if days_ago is None
        else datetime.now(UTC).date() - timedelta(days=days_ago),
        remote=False,
    )


def test_stale_postings_are_dropped_but_undated_ones_are_kept() -> None:
    """An unknown date is not an old one.

    ATS boards serve evergreen requisitions - a real digest carried a posting
    dated 2024-02-15 - but a handful of feeds give no date at all, and treating
    those as stale would delete a whole source rather than its old rows.
    """
    jobs = [
        _job_posted(5, "fresh"),
        _job_posted(400, "ancient"),
        _job_posted(None, "undated"),
    ]
    kept, dropped = _drop_stale(jobs, 90)
    assert dropped == 1
    assert {j.title for j in kept} == {"fresh", "undated"}


def test_a_zero_max_age_keeps_everything() -> None:
    jobs = [_job_posted(5), _job_posted(4000)]
    kept, dropped = _drop_stale(jobs, 0)
    assert dropped == 0
    assert len(kept) == 2


async def test_a_source_that_goes_quiet_is_reported(tmp_path: Path) -> None:
    """The alarm for this project's one recurring defect: a source that stops
    returning rows without raising, leaving a run that still exits 0."""
    worked = SourceReport(
        kind="greenhouse", slug="janestreet", label="Jane Street", count=228
    )
    async with Store(tmp_path / "s.db") as store:
        assert await _check_coverage([worked], store) == []

        silent = SourceReport(
            kind="greenhouse", slug="janestreet", label="Jane Street", count=0
        )
        quiet = await _check_coverage([silent], store)
    assert quiet == [("Jane Street", 228)]


async def test_a_source_that_never_worked_is_not_called_quiet(tmp_path: Path) -> None:
    """Otherwise the alarm fires every morning for an unconfigured board and
    trains the reader to ignore the line that matters."""
    never = SourceReport(kind="adzuna", slug="gb", label="Adzuna", count=0)
    async with Store(tmp_path / "s.db") as store:
        assert await _check_coverage([never], store) == []
        assert await _check_coverage([never], store) == []


async def test_a_failed_source_is_not_also_called_quiet(tmp_path: Path) -> None:
    """It already has its own line in the digest; reporting it twice buries
    the silent case among the loud ones."""
    ok = SourceReport(kind="lever", slug="prima", label="Prima", count=97)
    async with Store(tmp_path / "s.db") as store:
        await _check_coverage([ok], store)
        broke = SourceReport(
            kind="lever", slug="prima", label="Prima", count=0, error="HTTP 500"
        )
        assert await _check_coverage([broke], store) == []


async def test_two_entries_sharing_a_slug_keep_separate_histories(
    tmp_path: Path,
) -> None:
    """The Adzuna config has one entry per location, so kind and slug alone
    would make the two overwrite each other and hide a real outage."""
    london = SourceReport(kind="adzuna", slug="gb", label="Adzuna London", count=203)
    wide = SourceReport(kind="adzuna", slug="gb", label="Adzuna UK-wide", count=12)
    async with Store(tmp_path / "s.db") as store:
        await _check_coverage([london, wide], store)
        quiet = await _check_coverage(
            [
                SourceReport(kind="adzuna", slug="gb", label="Adzuna London", count=0),
                SourceReport(
                    kind="adzuna", slug="gb", label="Adzuna UK-wide", count=12
                ),
            ],
            store,
        )
    assert quiet == [("Adzuna London", 203)]


def test_an_ats_boards_dates_are_not_treated_as_freshness() -> None:
    """A requisition opened in 2019 and still on the board is still open.

    A uniform cutoff removed 145 of Jane Street's 228 live openings, and the
    digest filled the gap with agency reposts from aggregators, which are
    always dated yesterday. The date means different things on the two kinds
    of source, so only one of them can be aged out.
    """
    ancient = _job_posted(2000, "Ancient Req")
    board_job = ancient.model_copy(update={"source": "greenhouse"})
    feed_job = ancient.model_copy(
        update={"source": "linkedin", "url": "https://example.com/feed"}
    )

    kept, dropped = _drop_stale([board_job, feed_job], 90)
    assert dropped == 1
    assert [j.source for j in kept] == ["greenhouse"]


def test_a_suffixed_source_name_still_resolves_its_kind() -> None:
    """Adzuna reports itself as `adzuna:gb`; the kind is before the colon."""
    old_advert = _job_posted(400).model_copy(update={"source": "adzuna:gb"})
    _, dropped = _drop_stale([old_advert], 90)
    assert dropped == 1


def test_an_unknown_source_is_aged_out_rather_than_trusted() -> None:
    """A plugin that has not declared itself is likelier to be a job board
    than an employer's own careers page, and the safe default is to filter."""
    old_advert = _job_posted(400).model_copy(update={"source": "some-plugin"})
    _, dropped = _drop_stale([old_advert], 90)
    assert dropped == 1


async def test_a_fresh_copy_survives_when_its_longer_twin_is_stale(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merging keeps the longest description. Merging before the age filter
    let a stale survivor take its fresh twin down with it, and the role
    vanished from the digest."""
    stale_long = _job_posted(400, "Data Engineer").model_copy(
        update={
            "company": "Drax",
            "location": "London, UK",
            "url": "https://example.com/old",
            "description": "a much longer description of the same role",
        }
    )
    fresh_short = _job_posted(3, "Data Engineer").model_copy(
        update={
            "company": "Drax",
            "location": "South East London, London",
            "url": "https://example.com/new",
            "description": "short",
        }
    )

    async def fake_fetch_all(
        cfg: Config, store: Store
    ) -> tuple[list[SourceReport], list[Job]]:
        return [], [stale_long, fresh_short]

    monkeypatch.setattr("rolescan.pipeline.fetch_all", fake_fetch_all)
    config.profile.max_age_days = 90
    result = await run_scan(config, dry_run=True, check_llm=False)
    assert result.fetched == 2
    assert result.stale == 1
    assert result.unique == 1


# --- default mode (facts) is exercised end to end, not just unit-tested ----


@respx.mock
async def test_default_facts_mode_applies_profile_rules(tmp_path: Path) -> None:
    """`llm.mode` is left unset in this config, so this exercises the
    default (facts), not an override.

    One posting plainly requires more years than `profile.rules` allows and
    must be skipped, quoting the advert; a second states no years and is
    judged on fit_score alone. `FitScorer` is used directly for the first
    assertion group, because a rule skip is capped below `min_report_score`
    by design and so can never appear in `ScanResult.reportable` - there is
    no other way to inspect its `FitVerdict`. `run_scan` is then used end to
    end to confirm only the second posting reaches the digest.
    """
    years_quote = "5+ years of Python experience"
    over_years_description = (
        f"Python, trading, energy, day-ahead forecasting. Must have {years_quote}."
    )
    no_years_description = "Python, trading, energy, day-ahead forecasting."

    facts_over_years = {
        "level": {"value": "not_stated", "quote": ""},
        "years_required": {"value": 5, "quote": years_quote},
        "student_only": {"value": None, "quote": ""},
        "hard_bars": [],
        "field": {"value": None, "quote": ""},
        "fit_score": 80,
        "reason": "Strong Python and energy-market overlap.",
        "keywords_missing": [],
    }
    facts_no_years = {
        "level": {"value": "not_stated", "quote": ""},
        "years_required": {"value": None, "quote": ""},
        "student_only": {"value": None, "quote": ""},
        "hard_bars": [],
        "field": {"value": None, "quote": ""},
        "fit_score": 70,
        "reason": "Solid Python and trading overlap.",
        "keywords_missing": [],
    }

    request_bodies: list[dict[str, Any]] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_bodies.append(body)
        user = body["messages"][1]["content"]
        facts = facts_over_years if years_quote in user else facts_no_years
        return httpx.Response(200, json={"message": {"content": json.dumps(facts)}})

    respx.post("http://localhost:11434/api/chat").mock(side_effect=_respond)
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": OLLAMA_MODEL}]})
    )

    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": _KEYWORDS,
                "min_keyword_score": 18,
                "min_report_score": 55,
                "rules": {"max_years_required": 2},
            },
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "model": OLLAMA_MODEL,
                "extra_prompt": "SECRET-BLOCK",
            },
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )
    assert cfg.llm.mode == "facts", "the default, exercised here without an override"

    over_years = ScoredJob(
        job=Job(
            source="greenhouse",
            company="Acme",
            title="Experienced Energy Data Scientist",
            location="London, UK",
            url="https://boards.greenhouse.io/acme/jobs/1",
            description=over_years_description,
        ),
        keyword_score=40,
    )
    no_years = ScoredJob(
        job=Job(
            source="greenhouse",
            company="Acme",
            title="Graduate Energy Data Scientist",
            location="London, UK",
            url="https://boards.greenhouse.io/acme/jobs/2",
            description=no_years_description,
        ),
        keyword_score=40,
    )

    async with Store(tmp_path / "seen.db") as store:
        scorer = FitScorer(
            cfg.llm, cfg.profile, store, extra_prompt=cfg.llm.extra_prompt
        )
        scored = await scorer.score_all([over_years, no_years])

    assert scorer.errors == 0
    assert len(request_bodies) == 2, "one ollama call per posting"
    for body in request_bodies:
        assert body["format"] == PostingFacts.model_json_schema()
        system_text = body["messages"][0]["content"]
        assert "SECRET-BLOCK" not in system_text, "no extra_prompt seam in facts mode"
        assert "Scoring guidance" not in system_text, "judge-mode SYSTEM wording leaked"

    by_title = {s.job.title: s for s in scored}
    skipped = by_title["Experienced Energy Data Scientist"]
    assert skipped.fit is not None
    assert skipped.fit.verdict == Verdict.SKIP
    assert years_quote in skipped.fit.reason
    assert skipped.fit.fit_score <= cfg.profile.min_report_score - 1

    reported = by_title["Graduate Energy Data Scientist"]
    assert reported.fit is not None
    assert reported.fit.fit_score == 70

    # End to end: the rule-skipped posting never reaches the digest (its
    # score is capped below min_report_score by design), only the posting
    # judged on fit_score alone does. A SEPARATE db path here, not the one
    # the FitScorer pass above already wrote to: the facts cache is keyed on
    # content_hash, which is identical for these same two postings, so
    # reusing that db would let run_scan silently read the cache the earlier
    # pass filled rather than actually re-exercising the ollama path.
    cfg.output.db_path = tmp_path / "run_scan_seen.db"
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Experienced Energy Data Scientist",
                        "location": {"name": "London, UK"},
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
                        "content": over_years_description,
                        "updated_at": "2026-08-20T10:00:00Z",
                    },
                    {
                        "id": 2,
                        "title": "Graduate Energy Data Scientist",
                        "location": {"name": "London, UK"},
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/2",
                        "content": no_years_description,
                        "updated_at": "2026-08-20T10:00:00Z",
                    },
                ]
            },
        )
    )
    result = await run_scan(cfg, dry_run=True)
    assert len(request_bodies) == 4, (
        "run_scan must have made its own two fresh ollama calls against the "
        "new db, not silently read the FitScorer pass's cache"
    )
    assert [s.job.title for s in result.reportable] == [
        "Graduate Energy Data Scientist"
    ]


# --- postings a rule hid are carried to the digest, not dropped -------------


@respx.mock
async def test_rule_hidden_carries_what_the_rules_kept_out_of_the_digest(
    tmp_path: Path,
) -> None:
    """On 2026-10-06, 44 of 109 scored postings were hidden by `profile.rules`
    with no trace: a rule skip is capped below `min_report_score` and a block
    is dropped by `show_blocked: false`, so a mis-read advert or a rule bug was
    invisible. `rule_hidden` is what the digest lists them from.

    `min_report_score` is set low on purpose. A block is capped at 20, so at
    20 it clears the score gate and is removed only by `show_blocked: false` -
    the second route out of the digest, not just the first.
    """
    years_quote = "5+ years of Python experience"
    bar_quote = "UAE nationals only"
    descriptions = {
        1: f"Python, trading, energy, day-ahead forecasting. Must have {years_quote}.",
        2: f"Python, trading, energy, day-ahead forecasting. {bar_quote}.",
        3: "Python, trading, energy, day-ahead forecasting.",
    }
    titles = {
        1: "Experienced Energy Data Scientist",
        2: "Energy Data Scientist (Abu Dhabi)",
        3: "Graduate Energy Data Scientist",
    }
    not_stated: dict[str, object] = {
        "level": {"value": "not_stated", "quote": ""},
        "years_required": {"value": None, "quote": ""},
        "student_only": {"value": None, "quote": ""},
        "hard_bars": [],
        "field": {"value": None, "quote": ""},
        "fit_score": 80,
        "reason": "Strong Python and energy-market overlap.",
        "keywords_missing": [],
    }

    def _respond(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][1]["content"]
        facts = dict(not_stated)
        if years_quote in user:
            facts["years_required"] = {"value": 5, "quote": years_quote}
        elif bar_quote in user:
            facts["hard_bars"] = [{"kind": "nationality", "quote": bar_quote}]
        return httpx.Response(200, json={"message": {"content": json.dumps(facts)}})

    respx.post("http://localhost:11434/api/chat").mock(side_effect=_respond)
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": OLLAMA_MODEL}]})
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": jid,
                        "title": titles[jid],
                        "location": {"name": "London, UK"},
                        "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                        "content": descriptions[jid],
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                    for jid in (1, 2, 3)
                ]
            },
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": _KEYWORDS,
                "min_keyword_score": 18,
                "min_report_score": 20,
                "rules": {"max_years_required": 2},
            },
            "llm": {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL},
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                "show_blocked": False,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )

    result = await run_scan(cfg, dry_run=True)

    assert [s.job.title for s in result.reportable] == [titles[3]]
    assert result.hidden_blocked == 1, "the block cleared the gate first"
    hidden = {s.job.title: s.fit.rule for s in result.rule_hidden if s.fit}
    assert hidden == {titles[1]: "years", titles[2]: "hard_bar"}
    assert await _seen_uids(tmp_path / "seen.db") == set(), "--dry stays dry"


def test_rule_hidden_never_repeats_a_posting_the_digest_already_shows(
    energy_job: Job, gated_job: Job
) -> None:
    """At `min_report_score: 0` a rule skip's cap is 0, which clears the gate,
    so a rule-skipped posting can be reportable. It is then in the digest
    already and must not be listed a second time as hidden. A score-decided
    posting is never rule-hidden, whatever happened to it."""
    shown = _scored(energy_job, verdict=Verdict.SKIP, fit_score=0, rule="years")
    hidden = _scored(gated_job, verdict=Verdict.SKIP, fit_score=0, rule="level")
    unruled = _scored(energy_job.model_copy(update={"url": "https://x/9"}))
    assert _rule_hidden([shown, hidden, unruled], [shown], []) == [hidden]


@respx.mock
async def test_rule_hidden_carries_postings_a_hard_blockers_term_removed(
    tmp_path: Path,
) -> None:
    """A configured `hard_blockers` term forces `blocked` before (or without)
    the model, so with `show_blocked: false` the posting is gone and only a
    count said so. The owner added several nationality phrasings in one week;
    a term that matches the wrong thing is exactly the mistake the digest's
    "Hidden by your rules" section exists to surface, so the posting is
    carried there with the term. Run keyword-only, so `fit` is None."""
    bar = "UAE nationals only"
    content = "Python, trading, energy, day-ahead forecasting."
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": jid,
                        "title": title,
                        "location": {"name": "Abu Dhabi"},
                        "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                        "content": text,
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                    for jid, title, text in (
                        (1, "Energy Data Scientist", f"{content} {bar}."),
                        (2, "Graduate Energy Data Scientist", content),
                    )
                ]
            },
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": _KEYWORDS,
                "hard_blockers": ["uae nationals only"],
                "min_keyword_score": 18,
                "min_report_score": 10,
            },
            "llm": {"enabled": False},
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                "show_blocked": False,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )

    result = await run_scan(cfg, dry_run=True, check_llm=False)

    assert [s.job.title for s in result.reportable] == [
        "Graduate Energy Data Scientist"
    ]
    assert result.hidden_blocked == 1, "the existing count is unchanged"
    assert [(s.job.title, s.fit, s.blocker_hits) for s in result.rule_hidden] == [
        ("Energy Data Scientist", None, ["uae nationals only"])
    ]
    text = render_markdown(result)
    section = text[text.index("## Hidden by your rules") :]
    assert "**Your blocking terms (hard_blockers, excluded_locations)**" in section
    assert (
        "- **Acme** · [Energy Data Scientist](https://boards.greenhouse.io/acme/jobs/1)"
        ' · Abu Dhabi: blocked by "uae nationals only"'
    ) in section
    assert "1 hidden by your rules" in text and "1 blocked and hidden" in text


def test_rule_hidden_lists_a_term_blocked_posting_once(
    energy_job: Job, gated_job: Job
) -> None:
    """A posting both a rule and a `hard_blockers` term caught is one posting,
    a term-blocked posting the digest shows (`show_blocked: true`) is not
    hidden at all, and of the prefilter rejects only a term hit is."""
    both = _scored(gated_job, verdict=Verdict.BLOCKED, fit_score=20, rule="hard_bar")
    both = both.model_copy(update={"blocker_hits": ["uae national"]})
    term_only = _scored(energy_job).model_copy(update={"blocker_hits": ["sc"]})
    unscored = ScoredJob(
        job=energy_job.model_copy(update={"url": "https://x/8"}),
        blocker_hits=["dv"],
    )
    shown = ScoredJob(
        job=energy_job.model_copy(update={"url": "https://x/9"}),
        blocker_hits=["dv"],
    )
    judged = [both, term_only, unscored, shown]
    term_reject = ScoredJob(
        job=gated_job.model_copy(update={"url": "https://x/7"}),
        keyword_score=-11,
        blocker_hits=["uae national"],
    )
    low_reject = ScoredJob(job=gated_job.model_copy(update={"url": "https://x/6"}))
    assert _rule_hidden(judged, [shown], [term_reject, low_reject]) == [
        both,
        term_only,
        unscored,
        term_reject,
    ]


@respx.mock
async def test_rule_hidden_carries_a_posting_a_term_pushed_under_the_prefilter(
    tmp_path: Path,
) -> None:
    """The owner's config carries the same nationality and clearance phrases
    as 60-point `blockers` and as `hard_blockers`, so a term hit almost always
    takes a posting under `min_keyword_score`. It is then a prefilter reject:
    never judged, never ranked, recorded as seen - and, without this, never
    in the section that exists to show a wrong term. A reject with no term is
    low relevance, not a rule hide, and stays out."""
    bar = "UAE nationals only"
    relevant = "Python, trading, energy, day-ahead forecasting."
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": jid,
                        "title": title,
                        "location": {"name": "Abu Dhabi"},
                        "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                        "content": text,
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                    for jid, title, text in (
                        (1, "Energy Data Scientist", f"{relevant} {bar}."),
                        (2, "Receptionist", "Front desk cover."),
                        (3, "Graduate Energy Data Scientist", relevant),
                    )
                ]
            },
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": _KEYWORDS,
                "blockers": {"uae nationals only": 60},
                "hard_blockers": ["uae nationals only"],
                "min_keyword_score": 18,
                "min_report_score": 10,
            },
            "llm": {"enabled": False},
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                "show_blocked": False,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )

    result = await run_scan(cfg, dry_run=True, check_llm=False)

    assert result.prefiltered == 2, "both the term hit and the receptionist"
    assert [s.job.title for s in result.reportable] == [
        "Graduate Energy Data Scientist"
    ]
    [hidden] = result.rule_hidden
    assert hidden.job.title == "Energy Data Scientist"
    assert hidden.blocker_hits == ["uae nationals only"]
    gate = cfg.profile.min_keyword_score
    assert hidden.keyword_score < gate <= hidden.keyword_score + 60, (
        "the term's weight alone is what put it under the gate"
    )
    text = render_markdown(result)
    terms = text[text.index("**Your blocking terms") :]
    assert "[Energy Data Scientist](https://boards.greenhouse.io/acme/jobs/1)" in terms
    assert 'Abu Dhabi: blocked by "uae nationals only"' in terms
    assert "Receptionist" not in text
    assert "1 hidden by your rules" in text

    # Rejects are still recorded as seen on a real run, so the posting is
    # listed once, on the run that first sees it, and not every morning.
    first = await run_scan(cfg, check_llm=False)
    assert [s.job.title for s in first.rule_hidden] == ["Energy Data Scientist"]
    again = await run_scan(cfg, check_llm=False)
    assert again.rule_hidden == []
