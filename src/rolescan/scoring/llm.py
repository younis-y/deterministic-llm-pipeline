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
from rolescan.models import FitVerdict, ScoredJob
from rolescan.scoring.judges import TRIAGE_REASON, Judge, get_judge

if TYPE_CHECKING:
    from rolescan.store import Store

__all__ = ["FitScorer"]

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
        self._sem = asyncio.Semaphore(cfg.max_concurrent)
        self._calls = 0
        self._errors = 0
        self._first_error = ""
        self._judge: Judge | None = None

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
        job = scored.job

        if self.store is not None:
            cached = await self.store.get_verdict(job.content_hash, self.cfg.cache_days)
            if cached is not None and not self._stale_triage(cached):
                return scored.model_copy(update={"fit": cached, "llm_cached": True})

        async with self._sem:
            if self._calls >= self.cfg.max_calls_per_run:
                log.info(
                    "LLM call ceiling reached, leaving %r on keyword score", job.title
                )
                return scored
            self._calls += 1
            verdict = await self._call(scored)

        if self.store is not None:
            await self.store.put_verdict(job.content_hash, verdict)
        return scored.model_copy(update={"fit": verdict})

    async def _call(self, scored: ScoredJob) -> FitVerdict:
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
