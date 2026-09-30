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
from datetime import UTC, datetime, timedelta

from rolescan.config import Config, SourceEntry
from rolescan.dedup import merge_near_duplicates
from rolescan.http import Fetcher
from rolescan.models import Job, ScoredJob
from rolescan.scoring import (
    FitScorer,
    score_keywords,
    unusable_backend_reason,
    unusable_enricher_reason,
)
from rolescan.sources import get_source
from rolescan.sources.base import _REGISTRY, PostingCache, SourceSkipped
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
    enricher_unusable: str = ""
    """Why the configured enricher could not be used, or "".

    Deliberately a SEPARATE field from `llm_unusable`, not folded into it. A
    misspelt `llm.enricher` does not stop the judge from scoring - `FitScorer`
    just runs without the extra step - so it must not trip anything that reads
    `llm_unusable` as "the judge backend cannot run": `_record_assessed`'s
    `backend_broke`, the digest's "did not run at all" / "pre-scan backend
    check failed" notes, or the CLI's matching warnings. Those all behave
    exactly as if no enricher had been configured. This field exists purely so
    the digest and the CLI can print their own, distinct one-line note."""
    stale: int = 0
    """Postings dropped for being older than `profile.max_age_days`."""
    quiet_sources: list[tuple[str, int]] = field(default_factory=list)
    """Sources that returned nothing this run but have returned rows before,
    as (label, the most they have returned in their last five runs).

    This is the alarm for the one defect this project keeps producing: a
    source that stops working without raising, leaving a run that exits 0 and
    delivers less than it should. Nothing else notices - the digest still has
    content from the sources that do work."""
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


async def _preflight(cfg: Config) -> tuple[str, str]:
    """(why the configured judge cannot be used at all, why the configured
    enricher cannot be used), each "" if it can.

    Run before anything is fetched, so either kind of misconfiguration says
    so up front instead of surfacing later as a silently degraded digest.
    Returned as two SEPARATE strings on purpose (round 2 of this task's
    review folded the enricher reason into the backend one, which then read
    as a broken judge backend everywhere `llm_unusable` is consulted -
    `_record_assessed`'s `backend_broke`, the digest's "did not run at all"
    note, the CLI's matching warning - for a fault that does not stop
    scoring at all, only the extra step an enricher adds on top of it).
    """
    backend_reason = await unusable_backend_reason(cfg.llm)
    if backend_reason:
        log.warning(
            "LLM scoring is unavailable: %s. Postings will be ranked on "
            "keyword score alone.",
            backend_reason,
        )
    enricher_reason = unusable_enricher_reason(cfg.llm)
    if enricher_reason:
        log.warning(
            "enricher %r is not available (%s); postings are scored normally "
            "without the extra step",
            cfg.llm.enricher,
            enricher_reason,
        )
    return backend_reason, enricher_reason


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


def _source_key(report: SourceReport) -> str:
    """Identity for count history. Includes the label because two entries can
    share a kind and slug - the Adzuna config has one per location - and would
    otherwise overwrite each other's history."""
    return f"{report.kind}:{report.slug}:{report.label}"


def _ages_meaningfully(source: str) -> bool:
    """Whether an age cutoff means anything for the source that produced a job.

    `Job.source` is the source name, sometimes with a suffix ("adzuna:gb"), so
    the kind is everything before the first colon. An unknown kind is treated
    as an aggregator: a plugin that has not declared otherwise is far more
    likely to be a job board than an employer's own careers page.
    """
    cls = _REGISTRY.get(source.split(":", 1)[0])
    return cls is None or cls.dates_are_freshness


def _drop_stale(jobs: list[Job], max_age_days: int) -> tuple[list[Job], int]:
    """Remove stale postings from sources whose dates mean freshness.

    Two things are deliberately kept. A posting with no date, because unknown
    is not old and dropping undated rows deletes whole feeds rather than their
    stale entries. And everything from an employer's own ATS board, because
    there `posted` is when the requisition was opened, not when the advert went
    up - the listing's presence on the board is the freshness signal. Measured:
    a uniform 90-day cutoff removed 145 of Jane Street's 228 live openings and
    74 of IMC's 174, and replaced them in the digest with recruitment-agency
    reposts from the aggregators, which are always dated yesterday.
    """
    if max_age_days <= 0:
        return jobs, 0
    cutoff = datetime.now(UTC).date() - timedelta(days=max_age_days)
    kept = [
        j
        for j in jobs
        if j.posted is None
        or j.posted >= cutoff
        or not _ages_meaningfully(j.source)
    ]
    return kept, len(jobs) - len(kept)


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
    """LLM fit score, cached on the posting's content hash."""
    scorer = FitScorer(
        cfg.llm, cfg.profile, store, extra_prompt=cfg.llm.extra_prompt
    )
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


async def _check_coverage(
    reports: list[SourceReport], store: Store
) -> list[tuple[str, int]]:
    """Record what each source returned, and name the ones that went quiet.

    Quiet means: returned nothing this run, raised nothing, and has returned
    rows within its last five runs. A source that has never worked is not
    quiet, it is unconfigured, and saying so every morning would train the
    reader to ignore the line that matters.

    Sources that errored or skipped are excluded - those already have their
    own line in the digest, and reporting them twice buries the silent case
    among the loud ones.
    """
    quiet: list[tuple[str, int]] = []
    for report in reports:
        if not report.ok:
            continue
        if report.count == 0:
            previous = await store.source_high_water(_source_key(report))
            if previous > 0:
                quiet.append((report.label or report.slug, previous))
                log.warning(
                    "source %s returned nothing; it returned up to %d recently",
                    report.label or report.slug,
                    previous,
                )
    await store.record_source_counts(
        {_source_key(r): r.count for r in reports if r.ok}
    )
    return quiet


async def run_scan(
    cfg: Config, *, dry_run: bool = False, check_llm: bool = True
) -> ScanResult:
    """Run one scan.

    The body is the pipeline, one named stage per line. `check_llm` is False
    only when the caller has deliberately turned scoring off (`--no-llm`).
    """
    result = ScanResult(dry_run=dry_run, llm_backend=cfg.llm.backend)
    if check_llm:
        result.llm_unusable, result.enricher_unusable = await _preflight(cfg)

    # The store opens BEFORE fetching, not after: the structured source needs
    # the posting cache during fetch to skip detail pages whose sitemap lastmod
    # has not moved. Opening it afterwards would leave that cache write-only.
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        result.reports, raw = await fetch_all(cfg, store)
        result.fetched = len(raw)

        # Age filter first: merging keeps the longest description, so merging
        # first could let a stale survivor take its fresh twin down with it.
        fresh_raw, result.stale = _drop_stale(raw, cfg.profile.max_age_days)
        unique = merge_near_duplicates(deduplicate(fresh_raw))
        result.unique = len(unique)

        result.quiet_sources = await _check_coverage(result.reports, store)

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
