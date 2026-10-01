"""Stage two: LLM fit scoring and CV matching.

Backend-agnostic: whichever judge is configured returns `FitVerdict` as a
validated pydantic object rather than JSON coaxed out of prose. Both built-in
backends constrain sampling to the schema - the Anthropic one through
structured outputs, the Ollama one through `format` - so there is no
parse-retry loop and no defensive JSON repair on either path.

Three things keep this cheap:
  * the keyword prefilter, so only plausible roles get here at all
  * a persistent verdict cache keyed on the description text
  * a hard per-run call ceiling
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from rolescan.config import LLMConfig, ProfileConfig
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.scoring.enrich import Enricher, get_enricher
from rolescan.scoring.facts import (
    PostingFacts,
    resolve_field,
    resolve_level,
    verify_facts,
)
from rolescan.scoring.judges import TRIAGE_REASON, Judge, get_judge
from rolescan.scoring.rules import decide

if TYPE_CHECKING:
    from rolescan.store import Store

__all__ = ["SYSTEM_FACTS", "FitScorer", "cache_key", "final_key"]

#: A verdict worth doing extra work for. `skip` and `blocked` never reach an
#: enricher, whatever their score - and `decide` always caps their score just
#: under `min_report_score` anyway (see `rules.decide`), so the gate below
#: would exclude them regardless. Checked explicitly all the same: enrichment
#: is for postings the reader will act on, not an artifact of how the cap
#: happens to land.
_ENRICHABLE_VERDICTS = frozenset({Verdict.APPLY, Verdict.CONSIDER})

log = logging.getLogger(__name__)

# Three things about this prompt that are easy to "tidy" into a defect.
#
# It is the only instruction BOTH backends read. Ollama takes
# FitVerdict.model_json_schema() as a grammar and ignores the field
# descriptions entirely, so a rule that lives only in a description reaches
# the hosted backend alone. Anything the model must know belongs here.
#
# "Wrong seniority is skip, not blocked" is stated twice on purpose: once
# where seniority is discussed, once where `blocked` is defined. A 14b local
# model does not reliably carry a constraint across four paragraphs, and a wrong
# `blocked` costs the reader the role outright - dropped from the digest and
# recorded as seen. Repetition is cheap; one of these going missing is not.
#
# The band-to-verdict mechanism measured 0% score/verdict violations over the
# 25-posting benchmark. Do not rework it.
SYSTEM = """\
You screen job postings for one specific candidate. You are blunt and useful, \
not encouraging. A generous score wastes their week.

<candidate>
{summary}
</candidate>
{extra}
Scoring guidance:
- 85-100: strong match, apply today. Verdict: apply.
- 65-84: worth applying, some gaps. Verdict: apply or consider.
- 40-64: stretch or partial match, only if the pipeline is thin. Verdict: \
consider or skip.
- 0-39: not a good use of their time. Verdict: skip, or blocked but only for \
the hard bars below. A low score by itself is never a reason to use "blocked".

The bands overlap deliberately: "consider" is correct anywhere from 40 to 84.

If your score and verdict disagree with that table, change the score to fit \
the verdict, not the other way round. The verdict is the judgement call.

Be strict about seniority. A role wanting eight years is not a 70 for someone \
with one internship and a master's, however well the keywords line up. Score \
it on its own merits. The verdict is "skip", not "blocked": wrong seniority \
is not a structural bar.

Set verdict to "blocked" ONLY for hard structural bars the candidate cannot \
clear by being a better applicant: a nationality requirement such as an \
Emiratisation "UAE National" or "National Talent programme" posting, a \
security clearance, or a work authorisation they do not hold. A blocked role \
should also get a low fit_score. Being underqualified or overqualified is \
"skip", not "blocked".

Write "reason" as ONE sentence under 220 characters: the single fact that \
decides this match, not a summary of the posting. Twelve of these are read on \
a phone before work, so a second sentence costs more than it adds."""

USER = """\
<posting>
Title: {title}
Company: {company}
Location: {location}
Posted: {posted}

{description}
</posting>

Score this posting for the candidate."""

# Facts mode's user template. Ends by telling the model where its quotes come
# from, because a small model asked for "the exact text" will otherwise quote
# back a field label from this template itself (e.g. "Title:") rather than
# the posting's own words - which would then verify (the label really is in
# the rendered prompt) while saying nothing true about the posting.
USER_FACTS = """\
<posting>
Title: {title}
Company: {company}
Location: {location}
Posted: {posted}

{description}
</posting>

Extract the facts. Quote the posting's own text, never the field labels such \
as "Title:"."""

# Facts mode's system prompt. This is the only instruction the model reads for
# that mode, on both backends: Ollama takes PostingFacts.model_json_schema()
# as a grammar and enforces only the shape, ignoring every field description,
# so a rule that lives only in the schema reaches the hosted backend alone.
# Anything the model must know belongs here - see SYSTEM's comment above for
# the history behind that split. That is also why every enum value gets its
# own line below rather than relying on the schema's description strings.
#
# No extra_prompt and no CV vocabulary: unlike SYSTEM, this prompt has no seam
# for private context. The judgement it makes is skills and domain fit only;
# level, years, student status and eligibility are decided afterwards by
# `rolescan.scoring.rules.decide`, from the quoted facts extracted here, not
# by the model.
SYSTEM_FACTS = """\
You extract facts from a job posting for one specific candidate. You do not \
decide seniority fit, years fit, or eligibility - those are decided by code \
from the facts you extract below, not by you.

<candidate>
{summary}
</candidate>

For level, years_required, student_only and field: copy the exact text from \
the posting that states it, verbatim, into that fact's quote. Quote the \
shortest phrase that shows the fact, under 200 characters - never a whole \
paragraph. If the posting \
does not say, leave it not stated - never guess or infer a value. Encode "not \
stated" as: level = not_stated; years_required, student_only and field = \
null; and an empty quote in every case.

level - the career level the posting targets:
- graduate_entry: graduate schemes, entry-level, 0-1 years, internships or \
placements.
- junior: 1-2 years.
- mid: 3-5 years.
- senior: senior, or 5+ years.
- lead_principal: lead, principal, staff, head, or a manager of engineers.
- not_stated: the posting gives no level.

years_required - the MINIMUM years of experience the advert REQUIRES. For a \
range such as "3-5 years" use the low end, 3. Years described as "ideally", \
"preferred", or "nice to have" are NOT a requirement - leave years_required \
not stated.

student_only - true only if the posting restricts the role to current \
students; not stated otherwise.

field - the job's core work:
- data_engineering: building data pipelines, platforms, or ETL.
- ai_llm: building ML, LLM, or AI systems.
- data_science: modelling, statistics, or analytics research.
- analytics_bi: reporting, dashboards, or BI.
- software: general software engineering not centred on data or AI.
- other: anything else - sales, finance, consulting, operations, hospitality, \
and so on.

List hard_bars only for a bar the candidate cannot clear by being a better \
applicant, each with its own verbatim quote:
- nationality: a nationality-only requirement.
- clearance: a security clearance requirement.
- work_auth: a work authorisation requirement stated in the posting, such \
as "no visa sponsorship" or "must have the right to work in the UK". List it \
when the posting states it; work authorisation is checked by configured \
keywords, not by your judgement of the candidate.
- other: an explicit, mandatory eligibility requirement only - a licence, a \
legal status, or residency in a location - never experience, sector \
background or skills.
Being underqualified, overqualified, lacking a sector background, or in the \
wrong field is never a hard bar.

fit_score is 0-100 for skills and domain match ONLY. Ignore seniority, years, \
student status and eligibility completely when scoring - those are judged \
elsewhere, by code, not by you. A senior role in the candidate's domain \
scores on the domain overlap alone.

Write "reason" as ONE sentence under 220 characters: the single fact that \
most affects the skills/domain match.

List up to 8 keywords_missing: skills or tools the posting asks for that are \
not evidenced for this candidate."""


def cache_key(job: Job, mode: str, cfg: LLMConfig) -> str:
    """The verdict-cache key for this posting under this scoring mode.

    Facts mode and judge mode ask a different question of the same posting
    and can reach different answers, so a cached judge-mode verdict must
    never be handed back as a facts-mode result (or vice versa) just because
    the description hash matches. `facts-v2` is the scorer version: bumping
    it invalidates every facts-mode cache entry whenever the extraction
    prompt, schema, or cached PAYLOAD SHAPE changes in a way that would
    change the answer - v2 itself is the move from caching the post-rules
    FitVerdict to caching the verified PostingFacts underneath it, so a v1
    row (a FitVerdict) is never misread as the v2 shape (a PostingFacts).
    v3 adds the deterministic level pass (`resolve_level`), the 200-char quote
    cap and the narrower `other` bar prompt, so a v2 row would replay facts
    the current extraction would not produce. v4 (2.4.2) changes the title
    level table (no "manager"; "lead"/"staff" only before a role word; junior
    beats senior) and the work_auth prompt wording; cached facts hold the
    post-`resolve_level` level, so a v3 row would replay the old table.
    v5 (2.4.3) adds the deterministic field pass (`resolve_field`): cached
    facts hold the post-`resolve_field` field, so a v4 row would replay the
    model's (usually null) field.
    v6 (2.4.4) widens `_FIELD_WORDS` to match the natural inflections of the
    ai_llm/data_engineering/data_science role words ("Data Engineering",
    "AI Engineering", "Data Platforms", "AI Software Engineer"); a v5 row may
    hold a field the widened table would now resolve differently, so it must
    not be replayed.

    The facts key also names `cfg.backend` and `cfg.model`: facts extracted by
    one model must not be served (re-decided and enriched) for `cache_days`
    after the user switches to another, which is exactly the switch an
    evaluation of the two exists to inform. The judge-mode key is unchanged.
    """
    if mode == "facts":
        return f"{job.content_hash}:facts-v6:{cfg.backend}:{cfg.model}"
    return job.content_hash


def final_key(job: Job) -> str:
    """The cache key for the FINISHED verdict: after `decide`, and after any
    enricher.

    `cache_key` in facts mode deliberately holds the verified `PostingFacts`,
    not a verdict, so that a `rules` or `min_report_score` change is applied
    to every cached posting for free on the next run rather than replaying a
    verdict frozen under whatever was configured when the row was written
    (see `cache_key`'s docstring). That is the right cache for FitScorer's own
    re-decision, but it is the wrong shape for a consumer OUTSIDE scoring -
    application prep, or anything else that wants "what should be done about
    this posting" - which wants the decision itself, enriched subclass and
    all, not the facts it was made from. `final_key` is that separate row: a
    plain, mode-independent key written whenever a store exists, including on
    a cache hit, so it always reflects the most recently decided (and
    enriched) verdict regardless of whether the model was called this run.
    """
    return f"{job.content_hash}:final"


class FitScorer:
    """Scores postings through the configured judge, with caching and a spend
    ceiling. Everything expensive and easy to get wrong lives here rather than
    in the backends: the verdict cache, the per-run call limit, concurrency,
    ordering and error counting."""

    def __init__(
        self,
        cfg: LLMConfig,
        profile: ProfileConfig,
        store: Store | None = None,
        *,
        extra_prompt: str = "",
    ) -> None:
        self.cfg = cfg
        self.profile = profile
        self.store = store
        #: Appended to SYSTEM verbatim. The seam for anything this library has
        #: no business knowing about - a caller with private context to add
        #: supplies it here rather than teaching the public prompt its
        #: vocabulary. Empty by default, and an empty extra changes nothing.
        self.extra_prompt = extra_prompt
        #: The model the cache is read back with. A judge returning a subclass
        #: writes a payload FitVerdict refuses, and refusing it deletes the row
        #: and re-scores - a whole cache lost to a schema that was never wrong.
        self.verdict_model: type[FitVerdict] = FitVerdict
        self._sem = asyncio.Semaphore(cfg.max_concurrent)
        self._calls = 0
        self._errors = 0
        self._first_error = ""
        self._judge: Judge | None = None
        self._enricher: Enricher | None = None
        #: Whether `_get_enricher` has already resolved `cfg.enricher`. Built
        #: lazily and once, same reasoning as `_judge`: importing rolescan
        #: must not cost a plugin import, and `None` alone cannot distinguish
        #: "not built yet" from "no enricher configured".
        self._enricher_built = False
        #: The facts extracted for each posting scored in facts mode, keyed
        #: by job.url. Empty in judge mode. Kept for the evaluation, which
        #: reports per-field extraction accuracy against an answer key - a
        #: question `FitVerdict` alone cannot answer once `decide` has turned
        #: the facts into a verdict.
        self.last_facts: dict[str, PostingFacts] = {}
        if extra_prompt and cfg.mode == "facts" and not cfg.enricher:
            log.info(
                "llm.extra_prompt is set but not used: facts mode does not send "
                "it to the scoring call, and no llm.enricher is configured to "
                "receive it"
            )

    @property
    def enabled(self) -> bool:
        """Whether scoring will actually run.

        Asking for a key unconditionally disabled every local backend on any
        machine that had none — which is every machine the local backends
        exist for. LLMConfig already switches itself off when a hosted backend
        has no key, so by this point cfg.enabled is the whole answer.
        """
        return self.cfg.enabled

    @property
    def calls_made(self) -> int:
        return self._calls

    @property
    def errors(self) -> int:
        """Scoring calls that raised. Counted so a scan whose every call failed
        cannot be mistaken for a deliberate keyword-only run."""
        return self._errors

    @property
    def first_error(self) -> str:
        return self._first_error

    def _get_judge(self) -> Judge:
        """Built lazily so importing rolescan never costs an SDK import, and so
        the keyword-only path works with no backend installed at all."""
        if self._judge is None:
            self._judge = get_judge(self.cfg.backend, self.cfg)
        return self._judge

    def _get_enricher(self) -> Enricher | None:
        """The configured enricher, built at most once per scorer.

        Sets `self.verdict_model` to the enricher's `verdict_model` the
        moment one is configured, so a cache read of the FINAL verdict (see
        `final_key`) validates as the enriched subclass rather than refusing
        its extra fields.

        `get_enricher` raises for an unknown `cfg.enricher` name - the right
        thing for `_preflight` to fail fast on before a scan starts (see
        `unusable_enricher_reason`), but the wrong thing for a scorer already
        mid-run to do to every posting that reaches this method. The build
        is only ever attempted once (`_enricher_built` is set before the
        `try`, not after), so a bad name is logged once, not once per
        posting, and every posting afterwards is scored exactly as if no
        enricher had been configured at all.
        """
        if not self._enricher_built:
            self._enricher_built = True
            try:
                self._enricher = get_enricher(self.cfg)
            except Exception as e:
                log.warning(
                    "enricher %r is unusable, scoring without it: %s",
                    self.cfg.enricher,
                    e,
                )
                self._enricher = None
            if self._enricher is not None:
                self.verdict_model = self._enricher.verdict_model
        return self._enricher

    async def _maybe_enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        """Run the configured enricher, if any, for a verdict worth it.

        Guarded by the same semaphore and call ceiling as every other LLM
        call (`_call_judge`, `_call_facts`): an enricher is, by construction,
        a second model call, and letting it run outside `self._sem` would let
        a `max_concurrent=1` config still fire N enrichments at once, and
        skipping the `max_calls_per_run` check would let a run that hit its
        ceiling on facts calls alone go on to spend an unbounded number of
        enrichment calls on top of it.

        Any failure keeps the plain verdict and is logged - never counted
        against `self.errors`, and never a reason to drop the posting.
        Enrichment is additive, not load-bearing for whether a posting is
        reported at all.
        """
        enricher = self._get_enricher()
        if enricher is None:
            return verdict
        if verdict.verdict not in _ENRICHABLE_VERDICTS:
            return verdict
        if verdict.fit_score < self.profile.min_report_score:
            return verdict
        async with self._sem:
            if self._calls >= self.cfg.max_calls_per_run:
                log.info(
                    "LLM call ceiling reached, skipping enrichment for %r",
                    job.title,
                )
                return verdict
            self._calls += 1
            try:
                return await enricher.enrich(job, verdict)
            except Exception as e:
                log.warning("enrichment failed for %r: %s", job.title, e)
                return verdict

    def _system(self) -> str:
        return SYSTEM.format(
            summary=self.profile.summary or "(no summary configured)",
            extra=f"\n{self.extra_prompt}\n" if self.extra_prompt else "",
        )

    async def score_all(self, jobs: list[ScoredJob]) -> list[ScoredJob]:
        """Score every job, in ranked order so the budget buys the best ones.

        If the call ceiling bites, it bites on the weakest candidates. Anything
        not scored keeps its keyword score and is still reported.
        """
        if not self.enabled or not jobs:
            return jobs
        ordered = sorted(jobs, key=lambda s: -s.keyword_score)
        results = await asyncio.gather(
            *(self._score_one(s) for s in ordered), return_exceptions=True
        )
        out: list[ScoredJob] = []
        for original, result in zip(ordered, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("scoring failed for %s: %s", original.job.title, result)
                self._errors += 1
                if not self._first_error:
                    self._first_error = f"{type(result).__name__}: {result}"[:160]
                out.append(original)
            else:
                out.append(result)
        return out

    async def _score_one(self, scored: ScoredJob) -> ScoredJob:
        if self.cfg.mode == "facts":
            return await self._score_one_facts(scored)
        return await self._score_one_judge(scored)

    async def _score_one_judge(self, scored: ScoredJob) -> ScoredJob:
        job = scored.job
        key = cache_key(job, self.cfg.mode, self.cfg)

        if self.store is not None:
            cached = await self.store.get_verdict(
                key, self.cfg.cache_days, self.verdict_model
            )
            if cached is not None and not self._stale_triage(cached):
                return scored.model_copy(update={"fit": cached, "llm_cached": True})

        async with self._sem:
            if self._calls >= self.cfg.max_calls_per_run:
                log.info(
                    "LLM call ceiling reached, leaving %r on keyword score", job.title
                )
                return scored
            self._calls += 1
            verdict = await self._call_judge(scored)

        if self.store is not None:
            await self.store.put_verdict(key, verdict)
        return scored.model_copy(update={"fit": verdict})

    async def _score_one_facts(self, scored: ScoredJob) -> ScoredJob:
        """Facts mode's cache holds the VERIFIED `PostingFacts`, not the
        post-rules `FitVerdict` `decide()` makes from them.

        `rules` and `min_report_score` live in `self.profile`, read fresh on
        every call, so a cache hit is re-decided under whatever is configured
        NOW rather than replaying a verdict frozen under whatever was
        configured when the row was written. That is what lets a rules
        change (or a `min_report_score` change) apply to every posting
        already in cache at zero extra model calls, and what stops a rule
        skip's score cap from going stale if the gate moves after the row was
        written - caching the FitVerdict directly could do neither, since the
        rule that produced it would already be baked into the stored payload.
        """
        job = scored.job
        key = cache_key(job, self.cfg.mode, self.cfg)

        if self.store is not None:
            cached_facts = await self.store.get_verdict(
                key, self.cfg.cache_days, PostingFacts
            )
            if cached_facts is not None:
                self.last_facts[job.url] = cached_facts
                verdict = decide(
                    cached_facts, self.profile.rules, self.profile.min_report_score
                )
                verdict = await self._maybe_enrich(job, verdict)
                await self.store.put_verdict(final_key(job), verdict)
                return scored.model_copy(update={"fit": verdict, "llm_cached": True})

        async with self._sem:
            if self._calls >= self.cfg.max_calls_per_run:
                log.info(
                    "LLM call ceiling reached, leaving %r on keyword score", job.title
                )
                return scored
            self._calls += 1
            facts = await self._call_facts(scored)

        self.last_facts[job.url] = facts
        if self.store is not None:
            await self.store.put_verdict(key, facts)
        verdict = decide(facts, self.profile.rules, self.profile.min_report_score)
        verdict = await self._maybe_enrich(job, verdict)
        if self.store is not None:
            await self.store.put_verdict(final_key(job), verdict)
        return scored.model_copy(update={"fit": verdict})

    async def _call_facts(self, scored: ScoredJob) -> PostingFacts:
        job = scored.job
        user = USER_FACTS.format(
            title=job.title,
            company=job.company,
            location=job.location or "not stated",
            posted=job.posted.isoformat() if job.posted else "not stated",
            description=(
                job.description[: self.cfg.description_chars]
                or "(no description provided by the source)"
            ),
        )
        system = SYSTEM_FACTS.format(
            summary=self.profile.summary or "(no summary configured)"
        )
        judge = self._get_judge()
        # One call, always: Call 1 is already short, so the cascade - built to
        # skip generating prose for a role that will not clear the gate - buys
        # nothing here, and facts mode never runs it.
        verified = verify_facts(await judge.facts(system, user), job)
        return resolve_field(resolve_level(verified, job), job)

    async def _call_judge(self, scored: ScoredJob) -> FitVerdict:
        job = scored.job
        user = USER.format(
            title=job.title,
            company=job.company,
            location=job.location or "not stated",
            posted=job.posted.isoformat() if job.posted else "not stated",
            description=(
                job.description[: self.cfg.description_chars]
                or "(no description provided by the source)"
            ),
        )
        judge = self._get_judge()
        system = self._system()
        if self.cfg.cascade and judge.cheap_triage:
            first = await judge.triage(system, user)
            if first.fit_score < self.profile.min_report_score:
                return first
        return await judge.verdict(system, user)

    def _stale_triage(self, verdict: FitVerdict) -> bool:
        """Whether a cached triage stub has been brought into scope.

        A stub is only ever valid while it still scores below the gate. Lower
        `min_report_score` between runs and the stubs that now clear it have no
        reason and no blockers - so the digest would print a role
        with an empty justification and look broken. Cheaper to re-ask.
        """
        return (
            verdict.reason == TRIAGE_REASON
            and verdict.fit_score >= self.profile.min_report_score
        )
