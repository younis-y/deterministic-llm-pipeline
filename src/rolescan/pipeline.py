"""The scan pipeline.

    fetch all sources concurrently
      -> deduplicate within the run
      -> keyword score
      -> drop anything already reported
      -> prefilter to plausible roles
      -> LLM fit score and CV match (cached)
      -> rank, then record what was assessed

Everything here is orchestration. The judgement lives in scoring, the IO in
sources and store.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from rolescan.config import Config, ProfileConfig, SourceEntry
from rolescan.dedup import merge_near_duplicates
from rolescan.http import Fetcher
from rolescan.models import Job, ScoredJob, Verdict
from rolescan.scoring import (
    FitScorer,
    score_keywords,
    unusable_enricher_reason,
)
from rolescan.scoring.judges import backend_status
from rolescan.scoring.keyword import SYNTHETIC_PENALTIES
from rolescan.sources import get_source
from rolescan.sources.base import _REGISTRY, PostingCache, SourceSkipped
from rolescan.store import Store

__all__ = ["ScanResult", "SourceReport", "record_scan", "run_scan"]

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
    llm_model: str = ""
    """`llm.model` for this run, whether or not it ran (2.5.8)."""
    llm_model_digest: str = ""
    """The first 12 hex characters of the served model's digest, when the
    backend reports one (Ollama does, in `/api/tags`), else "" (2.5.8). A tag
    can be re-pulled under the same name; this is what says the weights
    behind the run's verdicts changed."""
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
    `llm_unusable` as "the judge backend cannot run": `assessed`'s
    `backend_broke`, the digest's "did not run at all" / "pre-scan backend
    check failed" notes, or the CLI's matching warnings. Those all behave
    exactly as if no enricher had been configured. This field exists purely so
    the digest and the CLI can print their own, distinct one-line note."""
    stale: int = 0
    """Postings dropped for being older than `profile.max_age_days`."""
    quiet_sources: list[tuple[str, int]] = field(default_factory=list)
    """Sources that returned nothing this run but have returned rows before,
    as (label, the most they have returned in the last 14 days).

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
    rule_hidden: list[ScoredJob] = field(default_factory=list)
    """Postings the owner's rules kept out of `reportable`: one of
    `profile.rules` fired (`fit.rule` is set), the model blocked it in judge
    mode (`fit.verdict` is blocked with no rule), or a `profile.hard_blockers`
    term or `excluded_locations` entry matched (`blocker_hits`), whichever way
    they left it. A rule skip's score is capped below `min_report_score`; a
    block, by the model's bar or by a term, is dropped by
    `output.show_blocked: false`; and a term hit is usually a prefilter
    reject, never judged at all, because the owner's config carries the same
    phrases as 60-point `blockers` too. Each posting appears once, however
    many of these caught it, and `fit` is None for one the model never scored.

    On 2026-10-06, 44 of 109 scored postings were hidden by rules with no
    trace, so a mis-read advert or a rule bug that skipped a good role was
    invisible; a wrong `hard_blockers` term (several nationality phrasings
    were added that week) is the same mistake by another route. The digest
    lists these, one line each, so a wrong skip can be spotted. A posting
    that is still reportable (a rule skip at `min_report_score: 0`, a block
    with `show_blocked: true`) is already in the digest and is not repeated.

    A prefilter reject is included when its terms are what put it under
    the gate: its keyword score plus the `blockers` weights of its
    `blocker_hits` clears `min_keyword_score`. (2.5.7: so is one whose
    weighted `blockers` terms alone did it, and one within
    `profile.hidden_gate_margin` of the gate, whatever put it there; both
    carry no rule, and the digest groups them as `blockers` and `gate`. A
    thin posting a `hard_blockers` term caught is listed too.) A reject that
    fails the gate by more than that either way is low relevance, not a rule
    hide. An `excluded_locations`
    entry carries no weight, so an irrelevant posting in an excluded location
    is not listed (about seven US postings a day on the owner's store), while
    a relevant one clears the gate, is judged, and is listed from there.
    Rejects are still recorded as seen, so a wrongly hidden posting is listed
    once, on the run it is first seen, and not again (a `--dry` run records
    nothing, so it lists it every time)."""
    deferred: list[ScoredJob] = field(default_factory=list)
    """Postings this run did not judge and did not record (2.5.7): past the
    LLM ceiling, past `output.max_roles`, or without a description. They are
    not `seen`, so the next run sees them again. Counted in the digest so a
    run that keeps deferring the same roles is visible, where before it
    buried them. A posting without a description is held back only for
    `output.thin_unread_after` runs: then it moves to `unread`."""
    unread: list[ScoredJob] = field(default_factory=list)
    """Postings that had no description on `output.thin_unread_after` runs
    (2.5.7), listed once under "Unread" and recorded with the reason
    `thin_unread`. "Hold, then list as unread": a source that
    never sends text (Workday with `details: false`, a structured page with no
    body, a harvested row released without text) would otherwise leave a
    posting deferred for ever, and a pile of them would crowd the judged ones
    out. Never touched on a dry run."""
    unread_after: int = 3
    """`output.thin_unread_after` for this run, so the digest can say "no text
    after 3 runs"."""
    to_record: list[tuple[ScoredJob, str]] = field(default_factory=list)
    """What a real run writes to `seen` once the digest is on disk (2.5.7),
    each posting with the reason it was assessed (see `assessed`). Empty on a
    dry run."""
    gate: int = 0
    """`profile.min_keyword_score` for this run (2.5.7), so the digest can say
    "scored 12 of 20" for a reject listed as just under it."""
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


async def _preflight(cfg: Config) -> tuple[str, str, str]:
    """(why the configured judge cannot be used at all, why the configured
    enricher cannot be used, the served model's digest), each "" if it can
    be used or is not known.

    Run before anything is fetched, so either kind of misconfiguration says
    so up front instead of surfacing later as a silently degraded digest.
    Returned as two SEPARATE strings on purpose (round 2 of this task's
    review folded the enricher reason into the backend one, which then read
    as a broken judge backend everywhere `llm_unusable` is consulted -
    `assessed`'s `backend_broke`, the digest's "did not run at all"
    note, the CLI's matching warning - for a fault that does not stop
    scoring at all, only the extra step an enricher adds on top of it).
    """
    status = await backend_status(cfg.llm)
    backend_reason = status.reason
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
    return backend_reason, enricher_reason, status.model_digest


async def _drop_already_handled(
    scored: list[ScoredJob], store: Store, *, touch: bool
) -> list[ScoredJob]:
    """Everything the reader has already dealt with, in one stage.

    Two different records mean the same thing to the reader. `seen` is what
    a previous scan reported; `dismissed` is what they explicitly threw away.
    The second cannot be done with the first: the same role arrives under a
    new uid from the next board that lists it, so it has to be matched on url
    as well.

    The caller counts both as already seen, before taking that total, so the
    stats line still reconciles: unique = already seen + filtered before
    scoring + whatever reached the scorer. `touch` refreshes `seen.last_seen`
    for the ones still listed (2.5.8); a dry run passes False and writes
    nothing.
    """
    fresh = await store.filter_new(scored, touch=touch)
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
        if j.posted is None or j.posted >= cutoff or not _ages_meaningfully(j.source)
    ]
    return kept, len(jobs) - len(kept)


def _prefilter(
    fresh: list[ScoredJob], gate: int
) -> tuple[list[ScoredJob], list[ScoredJob], list[ScoredJob]]:
    """Split by the keyword gate, holding back postings with nothing to read.

    The gate is the single biggest lever on cost, and the reason blocker terms
    are matched on word boundaries: a posting that falls below it is never
    judged, is recorded as seen, and never surfaces again.

    A posting whose description never arrived is gated on its title alone,
    which keeps 85% of what the full text keeps (measured 2026-10-07 on the
    owner's store: a graduate stream titled "IT, Tech and Data" scored 94 on
    its text and 0 on its title). So a thin reject is deferred, not rejected:
    the next run may have the text. A thin posting that clears the gate is
    judged as before; the model's quotes then come from the title.
    """
    candidates = [s for s in fresh if s.keyword_score >= gate]
    under = [s for s in fresh if s.keyword_score < gate]
    rejects = [s for s in under if s.job.description.strip()]
    thin = [
        s.model_copy(update={"deferred": "thin"})
        for s in under
        if not s.job.description.strip()
    ]
    return candidates, rejects, thin


async def _count_thin(
    thin: list[ScoredJob], store: Store, unread_after: int
) -> tuple[list[ScoredJob], list[ScoredJob]]:
    """Count this sighting of each text-less posting; split off the ones that
    have now gone `unread_after` runs without text.

    Returns (unread, still deferred). The count is what gives "deferred to the
    next run" an end: without it a source that never sends a description (a
    Workday board with `details: false`, a structured page with no body) holds
    the same postings back on every run, and hundreds of them crowd out the
    ones that can be judged. An unread posting is returned with `deferred`
    cleared, because it is no longer waiting for anything: it is listed once
    and recorded. Its `deferred` row is dropped by `record_scan`, with the
    rest, once the digest exists. Real runs only: a dry run must not count.
    """
    if not thin:
        return [], []
    counts = await store.bump_deferred([s.job.uid for s in thin], "thin")
    unread = [
        s.model_copy(update={"deferred": ""})
        for s in thin
        if counts.get(s.job.uid, 0) >= unread_after
    ]
    held = [s for s in thin if counts.get(s.job.uid, 0) < unread_after]
    return unread, held


async def _judge(
    candidates: list[ScoredJob], cfg: Config, store: Store
) -> tuple[list[ScoredJob], FitScorer]:
    """LLM fit score, cached on the posting's content hash."""
    scorer = FitScorer(cfg.llm, cfg.profile, store, extra_prompt=cfg.llm.extra_prompt)
    judged = await scorer.score_all(candidates)
    return judged, scorer


def assessed(
    rejects: list[ScoredJob], judged: list[ScoredJob], *, backend_broke: bool
) -> list[tuple[ScoredJob, str]]:
    """The postings that were actually assessed - and ONLY those.

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

    2.5.7: a `deferred` posting is never assessed, whatever marked it: the
    ceiling, the digest cap, or a missing description. Leaving it out is what
    lets it come round again.

    Each posting comes back with the reason it was assessed, which `seen`
    keeps so `rolescan unsee` and a reader of the table can tell a keyword
    reject from a judged skip: `prefilter` for a reject, then for a judged
    posting the `fit.rule` that fired, else `blocked` when a hard blocker
    matched, else `judged`.
    """
    kept: list[tuple[ScoredJob, str]] = [
        (s, "prefilter") for s in rejects if not s.deferred
    ]
    for s in judged:
        if s.deferred or (s.fit is None and backend_broke):
            continue
        if s.fit is not None and s.fit.rule:
            kept.append((s, s.fit.rule))
        elif s.blocker_hits:
            kept.append((s, "blocked"))
        else:
            kept.append((s, "judged"))
    return kept


def _hide_blocked(keep: list[ScoredJob]) -> tuple[list[ScoredJob], int]:
    """Remove blocked roles from the digest, and say how many that was.

    Logged as well as counted: these postings are recorded as seen once the
    digest is written, except a deferred one, an unjudged one when the
    backend broke, and everything on a dry run, none of which is recorded.
    So for the rest this is the only record that a specific role existed and
    was deleted on the strength of a configured term.
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


def _rank(
    judged: list[ScoredJob], cfg: Config
) -> tuple[list[ScoredJob], int, list[ScoredJob]]:
    """Gate on score, drop blocked roles if asked, sort, and cap.

    Returns the digest roles, the number the block removed from a digest
    they would otherwise have made, and the roles past the cap. The count is
    taken here rather than at the score gate, so a blocked posting that fell
    short of min_report_score is not counted: it was not the block that kept
    it out.

    2.5.7: the roles past `max_roles` are returned too, marked `digest_cap`,
    so the caller can keep them out of `seen`. Before, they were recorded
    with the shown ones and never came back: 15-17 reportable roles a day on
    5-6 Oct 2026, invisible. A role already deferred for another reason (the
    LLM ceiling) keeps that reason, so the digest's counts say what held it.
    """
    keep = [s for s in judged if s.score >= cfg.profile.min_report_score]
    hidden = 0
    if not cfg.output.show_blocked:
        keep, hidden = _hide_blocked(keep)
    keep.sort(key=lambda s: s.sort_key(), reverse=True)
    shown = keep[: cfg.output.max_roles]
    overflow = [
        s.model_copy(update={"deferred": s.deferred or "digest_cap"})
        for s in keep[cfg.output.max_roles :]
    ]
    return shown, hidden, overflow


def _caught_by_a_rule(item: ScoredJob) -> bool:
    """Whether a rule, the model's own block, or a blocking term caught it."""
    fit = item.fit
    model_caught = fit is not None and (
        fit.rule is not None or fit.verdict is Verdict.BLOCKED
    )
    return model_caught or bool(item.blocker_hits)


def _weight(s: ScoredJob, profile: ProfileConfig) -> int:
    """What its configured terms cost a posting: the `blockers` weight of every
    penalty it was actually charged (`keyword_penalties`), bar the synthetic
    agency and location ones. A `title_only_blockers` term found only in the
    body is not charged, so it gives nothing back."""
    return sum(
        profile.blockers.get(t, 0)
        for t in s.keyword_penalties
        if t not in SYNTHETIC_PENALTIES
    )


def _rule_hidden(
    judged: list[ScoredJob],
    reportable: list[ScoredJob],
    rejects: list[ScoredJob],
    profile: ProfileConfig,
    *,
    thin: Sequence[ScoredJob] = (),
) -> list[ScoredJob]:
    """The postings a rule or a blocking term caught that the digest does not
    show (see `ScanResult.rule_hidden`): judged ones, then prefilter rejects
    whose terms' `blockers` weights are what put them under the gate.

    2.5.7 adds rejects pushed under by weighted terms alone (listed under
    `blockers`) and rejects within `hidden_gate_margin` of the gate (listed
    under `gate`). `thin` is the postings deferred for having no description:
    they are not rejects, but one whose title matched a `hard_blockers` term
    is listed all the same, or it would be in no list at all. Only those: a
    thin near-gate or weighted-term posting, or one whose only hit is an
    `excluded_locations` entry (`location: ...`, nothing for the reader to
    check), would be listed again on every run until its text arrives. A
    thin posting that is listed is listed again on every run too, for the
    same reason: a deferred posting is not recorded.

    A weighted or near-gate reject is returned as a copy with `hidden_as`
    set (`"blockers"` / `"gate"`); the others are the objects passed in.

    Matched on identity, not equality: `_rank` filters and slices `judged`
    and `run_scan` copies only the postings it marks `digest_cap`, so the
    shown ones are the same objects, and two distinct postings can compare
    equal field for field. Reads only what is already in memory, so `--dry`
    stays dry.
    """
    shown = {id(s) for s in reportable}
    caught = [s for s in judged if id(s) not in shown and _caught_by_a_rule(s)]
    gate = profile.min_keyword_score

    # A reject with a term hit is listed under the terms group when the
    # weighted penalties it was charged are what put it under the gate, those
    # of the hit's own term or any other: the term is what the reader has to
    # check. It used to count only the hit's own charge, so a reject whose hit
    # was unweighted (a `hard_blockers` phrase with no `blockers` entry) and
    # whose weighted `director` and `10+ years` pushed it under was in no list.
    pushed_under = [
        s
        for s in rejects
        if s.blocker_hits and s.keyword_score + _weight(s, profile) >= gate
    ]
    thin_blocked = [
        s
        for s in thin
        if any(not hit.startswith("location: ") for hit in s.blocker_hits)
    ]

    weighted = [
        s
        for s in rejects
        if not s.blocker_hits
        and _weight(s, profile)
        and s.keyword_score + _weight(s, profile) >= gate
    ]
    listed = {id(s) for s in caught + pushed_under + weighted}
    near = [
        s
        for s in rejects
        if id(s) not in listed
        and profile.hidden_gate_margin
        and gate - profile.hidden_gate_margin <= s.keyword_score < gate
    ]
    # The group is decided here, where the weights are, and carried on a copy
    # (`hidden_as`): the digest only has the posting, and "pushed under by"
    # is a claim about what the weight did.
    return (
        caught
        + pushed_under
        + thin_blocked
        + [s.model_copy(update={"hidden_as": "blockers"}) for s in weighted]
        + [s.model_copy(update={"hidden_as": "gate"}) for s in near]
    )


async def _check_coverage(
    reports: list[SourceReport], store: Store, *, record: bool = True
) -> list[tuple[str, int]]:
    """Record what each source returned, and name the ones that went quiet.

    Quiet means: returned nothing this run, raised nothing, and has returned
    rows within the last 14 days. A source that has never worked is not
    quiet, it is unconfigured, and saying so every morning would train the
    reader to ignore the line that matters.

    Sources that errored or skipped are excluded - those already have their
    own line in the digest, and reporting them twice buries the silent case
    among the loud ones.

    `record=False` on a dry run (2.5.7): the alarm looks back over a window of
    recent runs, and a `--dry` run used to write its counts into that window,
    so a few of them taught the store that zero was normal for a source that
    had gone silent.
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
    if record:
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

    Since 2.5.7 this does NOT write `seen`. It returns what should be
    recorded in `result.to_record` (empty on a dry run), and the caller
    records it once the digest exists, with `record_scan`: `rolescan scan`
    does so right after `write_digest`. A caller that never records
    re-reports every posting on every run, so any new caller of `run_scan`
    must do the same.
    """
    result = ScanResult(
        dry_run=dry_run, llm_backend=cfg.llm.backend, llm_model=cfg.llm.model
    )
    if check_llm:
        (
            result.llm_unusable,
            result.enricher_unusable,
            result.llm_model_digest,
        ) = await _preflight(cfg)

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

        result.quiet_sources = await _check_coverage(
            result.reports, store, record=not dry_run
        )

        scored = [score_keywords(j, cfg.profile) for j in unique]

        fresh = await _drop_already_handled(scored, store, touch=not dry_run)
        result.already_seen = len(scored) - len(fresh)

        candidates, rejects, thin = _prefilter(fresh, cfg.profile.min_keyword_score)
        result.prefiltered = len(rejects)
        result.unread_after = cfg.output.thin_unread_after
        if not dry_run:
            result.unread, thin = await _count_thin(thin, store, result.unread_after)

        judged, scorer = await _judge(candidates, cfg, store)
        result.llm_calls = scorer.calls_made
        result.llm_cached = sum(1 for s in judged if s.llm_cached)
        result.llm_errors = scorer.errors
        result.llm_error_detail = scorer.first_error

        result.reportable, result.hidden_blocked, overflow = _rank(judged, cfg)
        cut = {s.job.uid: s for s in overflow}
        judged = [cut.get(s.job.uid, s) for s in judged]
        result.deferred = [s for s in judged if s.deferred] + thin
        result.gate = cfg.profile.min_keyword_score
        result.rule_hidden = _rule_hidden(
            judged, result.reportable, rejects, cfg.profile, thin=thin
        )

        if not dry_run:
            result.to_record = assessed(
                rejects,
                judged,
                backend_broke=bool(result.llm_unusable) or scorer.errors > 0,
            ) + [(s, "thin_unread") for s in result.unread]
    return result


async def record_scan(cfg: Config, result: ScanResult) -> int:
    """Write `result.to_record` to `seen` and return how many rows that was.

    What `cli.scan` does once the digest is on disk (2.5.7); a library caller
    of `run_scan` must call this or nothing is ever marked seen. Call it after
    the digest is written and before anything that can fail while rendering:
    a crash in between would leave a digest on disk whose postings are not
    recorded, and the next run would report them again.

    Also drops the deferral count of every posting it records (see
    `Store.bump_deferred`): a posting listed as unread, or one that arrived
    with its text after being held back, is no longer waiting for anything.

    A dry run has an empty `to_record`, so this opens nothing and returns 0.
    """
    if not result.to_record:
        return 0
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        await store.record_all(result.to_record)
        # A recorded posting needs no deferral count any more, whether it was
        # listed as unread or arrived with its text and was judged. Done here,
        # after the digest exists, so a failed write leaves the count in place.
        await store.forget_deferred(s.job.uid for s, _ in result.to_record)
    return len(result.to_record)
