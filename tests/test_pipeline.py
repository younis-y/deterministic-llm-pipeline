from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from conftest import OLLAMA_MODEL, mock_ollama
from rolescan.config import Config
from rolescan.digest import render_markdown
from rolescan.models import (
    Confidence,
    CVVariant,
    FitVerdict,
    Job,
    ScoredJob,
    Verdict,
)
from rolescan.pipeline import ScanResult, SourceReport, deduplicate, run_scan


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
        "cv_variant": CVVariant.ENERGY,
        "tailoring": ["Lead with the battery dispatch LP result."],
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
    assert "CV_EnergySystems-Modelling" in out
    assert "battery dispatch" in out
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
        tailoring=["Lead with the forecasting work."],
    )
    out = render_markdown(ScanResult(unique=1, reportable=[item]))
    assert "UAE National only" in out
    assert "**Send:**" not in out
    assert "Tailor it" not in out


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
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "claude-sonnet-5"}]})
    )
    cfg = _cfg(tmp_path, {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL})
    result = await run_scan(cfg)
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
    cfg = _cfg(tmp_path, {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL})
    result = await run_scan(cfg)
    assert len(result.reportable) == 1
    assert len(await _seen_uids(tmp_path / "seen.db")) == 1
    assert (await run_scan(cfg)).already_seen == 1


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
    cfg = _cfg(tmp_path, {"enabled": True, "backend": "ollama", "model": OLLAMA_MODEL})
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
