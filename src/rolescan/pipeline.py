"""The scan pipeline.

    fetch all sources concurrently
      -> deduplicate within the run
      -> keyword score
      -> drop anything already reported
      -> prefilter to plausible roles
      -> LLM fit score and CV match (cached)
      -> record and rank

Everything here is orchestration. The judgement lives in scoring, the IO in
sources and store.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from rolescan.config import Config, SourceEntry
from rolescan.http import Fetcher
from rolescan.models import Job, ScoredJob
from rolescan.scoring import (
    CVLibrary,
    FitScorer,
    score_keywords,
    unusable_backend_reason,
)
from rolescan.sources import get_source
from rolescan.sources.base import PostingCache, SourceSkipped
from rolescan.store import Store

__all__ = ["ScanResult", "SourceReport", "run_scan"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SourceReport:
    kind: str
    slug: str
    label: str
    count: int = 0
    error: str = ""
    skipped: bool = False
    """Declined to run (no credentials, unsupported region). Not a failure,
    but not a success either: nothing was tested."""

    @property
    def ok(self) -> bool:
        return not self.error and not self.skipped


@dataclass(slots=True)
class ScanResult:
    reports: list[SourceReport] = field(default_factory=list)
    fetched: int = 0
    unique: int = 0
    already_seen: int = 0
    prefiltered: int = 0
    llm_calls: int = 0
    llm_cached: int = 0
    llm_errors: int = 0
    llm_error_detail: str = ""
    llm_backend: str = ""
    """The backend scoring was configured to use, whether or not it ran.

    Carried so the digest can address the right failure. "A 401 here means
    ANTHROPIC_API_KEY is missing" is exactly wrong advice for someone whose
    local server returned HTTP 500, and the digest is the one place the end
    user reads."""
    llm_unusable: str = ""
    """Why the configured judge could not be used at all, or "".

    Distinct from `llm_errors`, which counts calls that were made and failed.
    This one means no call was ever attempted: wrong backend name, no key,
    SDK absent. Carried on the result so the digest can say it, because the
    person this matters to reads the 06:30 email, not the launchd log.

    Load-bearing, not only cosmetic: together with `llm_errors` it is what
    decides whether a posting is recorded as seen. Empty means "no judge was
    asked for", which is a deliberate keyword-only run and records normally;
    non-empty means "a judge was asked for and could not start", which must
    not bury postings it never looked at."""
    hidden_blocked: int = 0
    """Postings that scored high enough for the digest and were removed from
    it solely because they were blocked, with `output.show_blocked` false.

    Without this the deletion is invisible. A blocked posting is dropped from
    `keep` AND written to `seen`, so it can never resurface - and nothing in
    the digest, the CLI or the store said it had happened. That mattered less
    while only the LLM could block a posting, for a bar it had actually read.
    A configured term does it on a substring, over the model's explicit
    objection, so the reader needs a number: a blocker false positive and a
    quiet market are otherwise the same digest.

    Counted at the `show_blocked` filter, not at the score gate, so it means
    exactly "roles the block removed from a digest they would otherwise have
    made". A blocked posting that fell short of min_report_score was not kept
    out by the block and is not counted here."""
    reportable: list[ScoredJob] = field(default_factory=list)
    dry_run: bool = False

    @property
    def failed_sources(self) -> list[SourceReport]:
        return [r for r in self.reports if r.error and not r.skipped]

    @property
    def skipped_sources(self) -> list[SourceReport]:
        return [r for r in self.reports if r.skipped]


async def _fetch_one(
    entry: SourceEntry, fetcher: Fetcher, cache: PostingCache | None = None
) -> tuple[SourceReport, list[Job]]:
    report = SourceReport(kind=entry.kind, slug=entry.slug, label=entry.label)
    try:
        source = get_source(entry, fetcher, cache)
        jobs = await source.fetch()
    except SourceSkipped as e:
        report.skipped = True
        report.error = str(e)
        log.info("source %s/%s skipped: %s", entry.kind, entry.slug, e)
        return report, []
    except Exception as e:
        # One dead board must never take down the scan. This is the single most
        # important error boundary in the tool: forty sources means forty
        # chances for a 404 or a schema change.
        report.error = f"{type(e).__name__}: {e}"
        log.warning("source %s/%s failed: %s", entry.kind, entry.slug, report.error)
        return report, []
    report.count = len(jobs)
    return report, jobs


async def fetch_all(
    cfg: Config, cache: PostingCache | None = None
) -> tuple[list[SourceReport], list[Job]]:
    entries = cfg.enabled_sources
    if not entries:
        return [], []
    async with Fetcher(cfg.http) as fetcher:
        pairs = await asyncio.gather(*(_fetch_one(e, fetcher, cache) for e in entries))
    reports = [p[0] for p in pairs]
    jobs = [j for p in pairs for j in p[1]]
    return reports, jobs


def deduplicate(jobs: list[Job]) -> list[Job]:
    """Collapse the same role appearing on two boards.

    Keeps the richest copy: an aggregator hit and a direct ATS hit for one role
    should resolve to the ATS one, which has the full description and the real
    apply link.
    """
    best: dict[str, Job] = {}
    for job in jobs:
        current = best.get(job.uid)
        if current is None or len(job.description) > len(current.description):
            best[job.uid] = job
    return list(best.values())


async def _preflight(cfg: Config) -> str:
    """Why the configured judge cannot be used at all, or "".

    Run before anything is fetched, so a backend that cannot start says so
    instead of quietly degrading the whole digest to keyword scores.
    """
    reason = await unusable_backend_reason(cfg.llm)
    if reason:
        log.warning(
            "LLM scoring is unavailable: %s. Postings will be ranked on "
            "keyword score alone.",
            reason,
        )
    return reason


async def _drop_already_handled(
    scored: list[ScoredJob], store: Store
) -> list[ScoredJob]:
    """Everything the reader has already dealt with, in one stage.

    Two different records mean the same thing to the reader. `seen` is what
    a previous scan reported; `dismissed` is what they explicitly threw away.
    The second cannot be done with the first: the same role arrives under a
    new uid from the next board that lists it, so it has to be matched on url
    as well.

    The caller counts both as already seen, before taking that total, so the
    stats line still reconciles: unique = already seen + filtered before
    scoring + whatever reached the scorer.
    """
    fresh = await store.filter_new(scored)
    dismissed = await store.dismissed_urls()
    if dismissed:
        fresh = [s for s in fresh if s.job.url not in dismissed]
    return fresh


def _prefilter(
    fresh: list[ScoredJob], gate: int
) -> tuple[list[ScoredJob], list[ScoredJob]]:
    """Split into what the LLM will read and what it never will.

    The single biggest lever on cost, and the reason blocker terms are
    matched on word boundaries: a posting that falls below the gate is never
    judged, is recorded as seen, and never surfaces again.
    """
    candidates = [s for s in fresh if s.keyword_score >= gate]
    rejects = [s for s in fresh if s.keyword_score < gate]
    return candidates, rejects


async def _judge(
    candidates: list[ScoredJob], cfg: Config, store: Store
) -> tuple[list[ScoredJob], FitScorer]:
    """LLM fit score and CV match, cached on the posting's content hash."""
    cv_dir = cfg.resolve(cfg.profile.cv_dir) if cfg.profile.cv_dir else None
    cvs = CVLibrary.load(cv_dir)
    scorer = FitScorer(cfg.llm, cfg.profile, cvs, store)
    judged = await scorer.score_all(candidates)
    return judged, scorer


async def _record_assessed(
    store: Store,
    rejects: list[ScoredJob],
    judged: list[ScoredJob],
    *,
    backend_broke: bool,
) -> None:
    """Record the postings that were actually assessed - and ONLY those.

    Recording a posting writes it to `seen`, and `filter_new` then suppresses
    it forever. Doing that to a posting the intended judge never saw buries a
    real role on the strength of a verdict that was never reached. It has
    happened: a run whose backend could not start recorded forty new
    postings, every one of them unjudged, and they can never surface again.

    The line is between a judge that was SUPPOSED to work and did not, and no
    judge having been asked for. Scoring switched off on purpose is not a
    failure: those postings were judged, on keywords, which is that user's
    chosen judgement, and refusing to record them would repeat the same roles
    in every digest forever - the same silent failure this wave exists to
    end, just quieter.

    Prefiltered rejects are recorded either way: they were assessed, on
    keywords, and rejected on their merits.
    """
    recorded = list(rejects)
    recorded += [s for s in judged if s.fit is not None or not backend_broke]
    await store.record_all(recorded)


def _hide_blocked(keep: list[ScoredJob]) -> tuple[list[ScoredJob], int]:
    """Remove blocked roles from the digest, and say how many that was.

    Logged as well as counted: these postings have just been written to
    `seen`, so this is the only record that a specific role existed and was
    deleted on the strength of a configured term.
    """
    visible = [s for s in keep if not s.is_blocked]
    hidden = len(keep) - len(visible)
    for s in keep:
        if s.is_blocked:
            log.info(
                "blocked and hidden: %s at %s (%s)",
                s.job.title,
                s.job.company,
                ", ".join(s.blocker_hits) or "LLM verdict",
            )
    return visible, hidden


def _rank(judged: list[ScoredJob], cfg: Config) -> tuple[list[ScoredJob], int]:
    """Gate on score, drop blocked roles if asked, sort, and cap.

    Returns the digest roles and the number the block removed from a digest
    they would otherwise have made. That count is taken here rather than at
    the score gate, so a blocked posting that fell short of min_report_score
    is not counted: it was not the block that kept it out.
    """
    keep = [s for s in judged if s.score >= cfg.profile.min_report_score]
    hidden = 0
    if not cfg.output.show_blocked:
        keep, hidden = _hide_blocked(keep)
    keep.sort(key=lambda s: s.sort_key(), reverse=True)
    return keep[: cfg.output.max_roles], hidden


async def run_scan(
    cfg: Config, *, dry_run: bool = False, check_llm: bool = True
) -> ScanResult:
    """Run one scan.

    The body is the pipeline, one named stage per line. `check_llm` is False
    only when the caller has deliberately turned scoring off (`--no-llm`).
    """
    result = ScanResult(dry_run=dry_run, llm_backend=cfg.llm.backend)
    if check_llm:
        result.llm_unusable = await _preflight(cfg)

    # The store opens BEFORE fetching, not after: the structured source needs
    # the posting cache during fetch to skip detail pages whose sitemap lastmod
    # has not moved. Opening it afterwards would leave that cache write-only.
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        result.reports, raw = await fetch_all(cfg, store)
        result.fetched = len(raw)

        unique = deduplicate(raw)
        result.unique = len(unique)

        scored = [score_keywords(j, cfg.profile) for j in unique]

        fresh = await _drop_already_handled(scored, store)
        result.already_seen = len(scored) - len(fresh)

        candidates, rejects = _prefilter(fresh, cfg.profile.min_keyword_score)
        result.prefiltered = len(rejects)

        judged, scorer = await _judge(candidates, cfg, store)
        result.llm_calls = scorer.calls_made
        result.llm_cached = sum(1 for s in judged if s.llm_cached)
        result.llm_errors = scorer.errors
        result.llm_error_detail = scorer.first_error

        if not dry_run:
            await _record_assessed(
                store,
                rejects,
                judged,
                backend_broke=bool(result.llm_unusable) or scorer.errors > 0,
            )

    result.reportable, result.hidden_blocked = _rank(judged, cfg)
    return result
