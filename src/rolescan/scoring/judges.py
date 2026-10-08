"""LLM backends, as plugins.

Every source except Adzuna is a public endpoint and keyword scoring is pure
Python, so the LLM stage is the only part of rolescan that ever needed a
credential. Making it a plugin keeps the tool useful with none: run the
keyword prefilter alone, point it at a local model, or supply an API key —
the pipeline, the cache, the digest and the CV matching are identical either
way.

A judge does one thing: turn a rendered prompt into a validated FitVerdict.
Everything expensive and easy to get wrong — the verdict cache, the spend
ceiling, concurrency, ordering, error counting — stays in FitScorer, so a new
backend is one method rather than a fork of the scorer.

Registration mirrors `rolescan.sources`: subclass, decorate with @register, or
ship one from another package under the `rolescan.judges` entry-point group.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Annotated, Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rolescan.config import LLMConfig
from rolescan.models import Confidence, FitVerdict, Verdict
from rolescan.scoring.facts import PostingFacts

__all__ = [
    "BackendStatus",
    "Judge",
    "OllamaUsage",
    "PromptTruncatedError",
    "available_judges",
    "backend_status",
    "get_judge",
    "ollama_options",
    "register",
    "served_model_digest",
    "unusable_backend_reason",
]

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Judge]] = {}

#: How long a preflight liveness check is allowed to take. Short on purpose:
#: this runs before every scan, not just when something is already wrong, so
#: it must never become a new way for the tool to hang. `LLMConfig.timeout`
#: is for a chat completion and is far too long for this.
PREFLIGHT_TIMEOUT = 3.0

#: The longer timeout for the one re-read of Ollama's model list when the
#: preflight could not say which weights are being served (2.5.8). The cache
#: key needs the answer, and a server busy loading a large model can miss the
#: 3 s probe and still answer in 10.
IDENTITY_TIMEOUT = 10.0

#: Seconds before the one retry of an Ollama call that timed out or got a 5xx.
#: A runner restarting mid-call, or a model being reloaded under memory
#: pressure, answers on the second try; a server that is really down fails
#: twice and is counted once.
RETRY_DELAY = 2.0

#: The fewest prompt tokens per character a real prompt has ever measured.
#: 2026-10-07, 27 calls to qwen2.5:14b with `prompt_eval_count`: 0.23 tokens
#: per character of instructions and examples, 0.18 per character of advert.
#: A prompt therefore holds at least one token per 6 characters; Ollama
#: reporting fewer means it cut the prompt (forced to num_ctx 4,096, it
#: evaluated 2,050 tokens of a 7,290-token prompt and still returned facts).
_CHARS_PER_TOKEN_CEILING = 6

#: Prompt plus answer this close to `num_ctx` is a prompt that may have been
#: cut to fit, or an answer cut off before its closing brace.
_CONTEXT_HEADROOM = 0.97


class PromptTruncatedError(RuntimeError):
    """Ollama read less of the prompt than was sent, or filled its window."""


@dataclass(frozen=True, slots=True)
class OllamaUsage:
    """The token counts Ollama reports on every `/api/chat` answer."""

    prompt_tokens: int
    output_tokens: int


def ollama_options(cfg: LLMConfig, *, num_predict: int | None = None) -> dict[str, Any]:
    """The `options` block for every Ollama call rolescan makes (2.5.8).

    One helper so no call can leave `num_ctx` out: without it Ollama sizes
    the context window from the machine's memory, and a smaller machine cut
    the facts prompt in half with no error. A plugin that talks to Ollama
    itself (an enricher, a cover-letter writer) builds its options here too,
    passing its own `num_predict`.
    """
    return {
        "num_ctx": cfg.num_ctx,
        "num_predict": cfg.max_tokens if num_predict is None else num_predict,
        "temperature": cfg.temperature,
    }


def ollama_usage(
    body: object,
    *,
    prompt_chars: int,
    num_ctx: int,
    check: bool = True,
    window: int | None = None,
) -> OllamaUsage | None:
    """The usage an `/api/chat` answer reports; raises if the prompt was cut.

    Returns None when the server reports no counts (an old Ollama, or a
    proxy in between): nothing can be checked then. Raises
    `PromptTruncatedError` when the server evaluated fewer prompt tokens
    than `prompt_chars` can hold (it dropped part of the prompt to fit), or
    when prompt and answer together reached `_CONTEXT_HEADROOM` of the
    window (it was full). Either way the answer was made without the whole
    prompt, and its facts must not be trusted or cached.

    The window is `num_ctx`, or the model's own `window` when the preflight
    read one and it is smaller: Ollama never runs a model past its own
    window, so a server that quietly sized the request down to it is checked
    against the window it really used.

    `check=False` (`llm.check_truncation: false`) returns the counts without
    judging them: the way out for a server whose `prompt_eval_count` does not
    count the whole prompt.
    """
    if not isinstance(body, dict):
        return None
    prompt, output = body.get("prompt_eval_count"), body.get("eval_count")
    if not isinstance(prompt, int) or not isinstance(output, int):
        return None
    if not check:
        return OllamaUsage(prompt_tokens=prompt, output_tokens=output)
    floor = prompt_chars // _CHARS_PER_TOKEN_CEILING
    if prompt < floor:
        msg = (
            f"ollama read {prompt} prompt tokens of a prompt that holds at least "
            f"{floor}: it cut the prompt to fit its context window. Raise "
            f"llm.num_ctx (now {num_ctx}), or check the server's own limit. "
            "If this server counts only the tokens it did not already have "
            "cached, set llm.check_truncation: false."
        )
        raise PromptTruncatedError(msg)
    limit = num_ctx if window is None else min(num_ctx, window)
    if prompt + output >= _CONTEXT_HEADROOM * limit:
        if limit < num_ctx:
            source = f"the model's own, below llm.num_ctx ({num_ctx})"
            fix = "Lower llm.num_ctx to it, or pick a larger model."
        else:
            source, fix = "llm.num_ctx", "Raise llm.num_ctx."
        msg = (
            f"ollama's prompt and answer filled {prompt + output} of the "
            f"{limit}-token context window ({source}): the prompt may have "
            f"been cut, or the answer stopped short. {fix}"
        )
        raise PromptTruncatedError(msg)
    return OllamaUsage(prompt_tokens=prompt, output_tokens=output)


def _model_window(info: dict[str, Any]) -> int | None:
    """The language model's context window from `/api/show`'s `model_info`.

    Keys are `<architecture>.context_length`, and `general.architecture` names
    which architecture the model is: read that one first, since `model_info`
    of a multimodal model can carry a second, unrelated `*.context_length`
    ahead of it. Without an architecture, or without that key, the first
    `*.context_length` is the best guess; None when there is none.
    """
    architecture = info.get("general.architecture")
    if isinstance(architecture, str):
        named = info.get(f"{architecture}.context_length")
        if isinstance(named, int) and not isinstance(named, bool):
            return named
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


class Judge(ABC):
    """Renders a prompt into a FitVerdict, however it likes."""

    name: ClassVar[str] = ""
    #: True when the backend talks to a hosted API that authenticates. Read by
    #: LLMConfig, so a local backend is not disabled for want of a key it never
    #: needed.
    needs_api_key: ClassVar[bool] = False
    #: Env var the no-key message tells the user to set. Only meaningful when
    #: `needs_api_key` is True; a backend that needs no key leaves it blank.
    api_key_env: ClassVar[str] = ""
    #: Import name of the SDK this backend needs, if any. Declared rather than
    #: discovered because the import itself is deliberately lazy, and a missing
    #: package should be one line at startup, not N identical scoring errors
    #: after the scan has already spent its fetch budget.
    requires_module: ClassVar[str] = ""
    #: One line, shown by `rolescan backends`.
    description: ClassVar[str] = ""
    #: The model's own context window in tokens, set by `preflight` when the
    #: backend can tell (2.5.8), else None. `FitScorer` hands the window its
    #: preflight read to the judge it builds; a backend that checks its
    #: answers' token counts checks them against it.
    context_window: int | None = None

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        #: The first 12 hex characters of the served model's digest, set by
        #: `preflight` when the backend can tell (2.5.8), else "". A tag such
        #: as `qwen2.5:14b` can be re-pushed upstream and pulled again; the
        #: digest is what says the weights changed.
        self.model_digest = ""

    #: True when `triage` is genuinely cheaper than `verdict`. False by
    #: default, and FitScorer will not run the cascade against a backend that
    #: leaves it False - the default `triage` is a full call, so cascading on
    #: one would buy nothing and bill twice.
    cheap_triage: ClassVar[bool] = False

    @abstractmethod
    async def verdict(self, system: str, user: str) -> FitVerdict:
        """One posting, one judgement. Raise on failure; FitScorer counts it."""

    async def facts(self, system: str, user: str) -> PostingFacts:
        """Extract quoted facts from one posting, for `llm.mode: facts`.

        The default raises: a judge that only speaks the old single-call
        protocol - built-in or third-party - explains itself the moment
        facts mode is selected, rather than failing with an AttributeError
        deep inside FitScorer once a scan is already under way.
        """
        msg = f"{type(self).__name__} does not implement facts mode"
        raise NotImplementedError(msg)

    async def triage(self, system: str, user: str) -> FitVerdict:
        """A first pass that only has to be right about the score.

        The digest prints `reason` and `blockers` only for roles
        that clear `min_report_score`; everything below it pays to generate
        prose no one reads. Generating that prose is ~85% of a local call.

        The default is the full call, which is correct and no cheaper - hence
        `cheap_triage`. A backend that can constrain output to a subset of the
        schema overrides both.
        """
        return await self.verdict(system, user)

    async def preflight(self) -> str:
        """Why this backend cannot be reached right now, or "" if it can.

        Distinct from a static cause like a missing key or SDK: this is a
        liveness fact, checked once before the scan starts, not at import
        time. The default is "nothing to check" — true of a hosted API,
        whose failure modes (bad key, no network) already surface at the
        first call. A backend that can be silently unreachable, like a local
        server, overrides this.
        """
        return ""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.cfg.model}>"


def register(cls: type[Judge]) -> type[Judge]:
    if not cls.name:
        msg = f"{cls.__name__} must set a class-level `name`"
        raise ValueError(msg)
    _REGISTRY[cls.name] = cls
    return cls


def load_plugins() -> None:
    """Pull in any third-party judges advertising the entry-point group."""
    try:
        eps = entry_points(group="rolescan.judges")
    except Exception:  # pragma: no cover - importlib differences across runtimes
        return
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:  # pragma: no cover - a broken plugin is not fatal
            log.warning("could not load judge plugin %s: %s", ep.name, e)
            continue
        if isinstance(obj, type) and issubclass(obj, Judge):
            register(obj)


def available_judges() -> dict[str, type[Judge]]:
    return dict(_REGISTRY)


def get_judge(name: str, cfg: LLMConfig) -> Judge:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        msg = f"unknown llm backend {name!r}. Registered: {known}"
        raise KeyError(msg) from None
    return cls(cfg)


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # find_spec raises rather than returning None when a parent package is
        # missing or the name is malformed. Either way it cannot be imported.
        return False


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """What the pre-scan check learned about the configured judge (2.5.8)."""

    reason: str = ""
    """Why it cannot score anything, or "" (see `unusable_backend_reason`)."""
    model_digest: str = ""
    """The served model's digest prefix, when the backend reports one."""
    context_window: int | None = None
    """The model's own context window in tokens, when the backend reports one
    (Ollama does, in `/api/show`), else None. Kept when it is the reason the
    backend cannot be used: a window below `llm.num_ctx` stops the LLM stage
    of a scan outright, where every other reason is a probe that scoring
    still tries past."""


async def backend_status(cfg: LLMConfig) -> BackendStatus:
    """`unusable_backend_reason`, plus the model digest the check read.

    One preflight, so the digest costs no second request: Ollama's
    `/api/tags`, which the liveness check already fetches, carries it. The
    model's context window comes from the same check (`/api/show`).
    """
    reason, judge = await _check_backend(cfg)
    digest = getattr(judge, "model_digest", "") if judge is not None else ""
    window = getattr(judge, "context_window", None) if judge is not None else None
    return BackendStatus(
        reason=reason,
        model_digest=digest if not reason else "",
        context_window=window if isinstance(window, int) else None,
    )


async def unusable_backend_reason(cfg: LLMConfig) -> str:
    """Why the configured judge cannot score anything, or "" if it can.

    See `_check_backend`; `backend_status` returns the same reason with the
    model digest beside it.
    """
    return (await backend_status(cfg)).reason


async def _check_backend(cfg: LLMConfig) -> tuple[str, Judge | None]:
    """(why the configured judge cannot score anything, or "" if it can; the
    judge that was checked, or None when it could not be built).

    Checked before the scan runs, because every way this goes wrong used to go
    wrong silently. A misnamed backend, a revoked key, an SDK that was never
    installed, a local server that is not running: each one leaves
    `FitScorer` handing back keyword scores while the run reports success, and
    a keyword-only digest against an LLM-calibrated `min_report_score` is
    empty by construction. The digest then says "nothing new worth your time"
    about a working market.

    Gated on `cfg.wants_scoring`, not `cfg.enabled`: `LLMConfig` switches
    itself off when a hosted backend has no key, which is the single most
    common cause and the one that most needs saying out loud. A config that
    asked for no scoring in the first place gets nothing to read.

    The first three checks are static — name, key, SDK — and cost nothing.
    The last, `Judge.preflight()`, is a liveness fact and needs the network,
    so it only runs once the static causes are ruled out. Async for that
    reason alone: this is the only check here that ever awaits anything.
    """
    if not cfg.wants_scoring:
        return "", None
    cls = _REGISTRY.get(cfg.backend)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        return (
            f"llm.backend {cfg.backend!r} is not a registered backend "
            f"(registered: {known})"
        ), None
    if cls.needs_api_key and not cfg.api_key:
        env = cls.api_key_env or "the appropriate environment variable"
        return (
            f"backend {cfg.backend!r} is configured but no API key is visible "
            f"(export {env}, or set llm.api_key in the config)"
        ), None
    if cls.requires_module and not _importable(cls.requires_module):
        return (
            f"backend {cfg.backend!r} needs the {cls.requires_module} package, "
            f"which is not installed (pip install 'rolescan[{cfg.backend}]')"
        ), None
    try:
        judge = cls(cfg)
        return await judge.preflight(), judge
    except Exception as e:
        # A judge is a plugin and `cls(cfg)` now runs at startup, so a broken
        # __init__ or a preflight that raises instead of returning would take
        # the whole scan down before a single source was fetched: no digest,
        # no email, a traceback in a launchd log nobody reads. Treated as
        # "this backend is unusable", which is what it is.
        log.warning("preflight for backend %s raised: %s", cfg.backend, e)
        reason = (
            f"backend {cfg.backend!r} could not be checked: {type(e).__name__}: {e}"
        )
        return reason, None


# ---------------------------------------------------------------------------
# built-in backends
# ---------------------------------------------------------------------------


@register
class AnthropicJudge(Judge):
    """The Claude API, via structured outputs.

    The schema is enforced by the API, so there is no parse-retry loop and no
    defensive JSON repair.
    """

    name = "anthropic"
    needs_api_key = True
    api_key_env = "ANTHROPIC_API_KEY"
    requires_module = "anthropic"
    description = "Claude API. Best quality. Needs ANTHROPIC_API_KEY."
    cheap_triage = True

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Built lazily so importing rolescan never costs an SDK import, and so
        the keyword-only path works with anthropic absent."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.cfg.api_key)
        return self._client

    @staticmethod
    def _cached_system(system: str) -> list[dict[str, Any]]:
        """Wrap the system prompt as one cached block.

        Every Anthropic call - verdict, triage, facts - reads the same
        candidate summary, so caching it once buys a discount on every call
        after the first. Haiku 4.5 only caches prefixes of >= 4,096 tokens
        and silently ignores smaller ones, so a short prompt (facts mode's)
        is expected not to cache - this is still correct to send, and the
        eval reports whether it actually does.
        """
        return [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    async def verdict(self, system: str, user: str) -> FitVerdict:
        response = await self._get_client().messages.parse(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=self._cached_system(system),
            messages=[{"role": "user", "content": user}],
            output_format=FitVerdict,
        )
        parsed = response.parsed_output
        if not isinstance(parsed, FitVerdict):  # pragma: no cover - enforced upstream
            msg = f"model returned {type(parsed).__name__}, expected FitVerdict"
            raise RuntimeError(msg)
        return _without_rule(parsed)

    async def facts(self, system: str, user: str) -> PostingFacts:
        response = await self._get_client().messages.parse(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=self._cached_system(system),
            messages=[{"role": "user", "content": user}],
            output_format=PostingFacts,
        )
        parsed = response.parsed_output
        if parsed is None:
            stop = getattr(response, "stop_reason", None)
            msg = f"claude returned no posting facts (stop_reason={stop!r})"
            raise RuntimeError(msg)
        if not isinstance(parsed, PostingFacts):  # pragma: no cover - enforced upstream
            msg = f"model returned {type(parsed).__name__}, expected PostingFacts"
            raise RuntimeError(msg)
        return parsed

    async def triage(self, system: str, user: str) -> FitVerdict:
        """Score and route without paying for prose the digest will not print.

        Same contract as the Ollama triage: only the fields the gate needs,
        under a small output cap, returned as a FitVerdict stub marked with
        TRIAGE_REASON.
        """
        response = await self._get_client().messages.parse(
            model=self.cfg.model,
            max_tokens=TRIAGE_MAX_TOKENS,
            system=self._cached_system(system),
            messages=[{"role": "user", "content": user}],
            output_format=_TriageOutput,
        )
        parsed = response.parsed_output
        if parsed is None:
            stop = getattr(response, "stop_reason", None)
            msg = f"claude returned no triage verdict (stop_reason={stop!r})"
            raise RuntimeError(msg)
        return FitVerdict(
            fit_score=parsed.fit_score,
            verdict=parsed.verdict,
            confidence=parsed.confidence,
            reason=TRIAGE_REASON,
        )


def _without_rule(verdict: FitVerdict) -> FitVerdict:
    """`verdict` with `rule` cleared: only `rules.decide` names a rule.

    `rule` is kept out of the schema both backends sample against, but
    validation still accepts the key, and a model that echoes it anyway would
    mark the posting rule-hidden in the digest on its own say-so. Defence in
    depth behind the schema, not a replacement for it.
    """
    if verdict.rule is None:
        return verdict
    return verdict.model_copy(update={"rule": None})


#: Marks a verdict produced by the triage pass, which was never asked for a
#: reason. It is a sentinel, not prose for a reader: FitScorer matches on it to
#: spot a cached stub that a lowered threshold has brought into scope.
TRIAGE_REASON = "(triage pass: below the reporting threshold, no detail requested)"

#: Everything needed to build a valid FitVerdict and decide the gate, minus the
#: two generated fields. `confidence` is an enum costing a handful of tokens
#: and carrying real information, so it is asked for rather than invented.
_TRIAGE_FIELDS = ("fit_score", "verdict", "confidence")


def triage_schema() -> dict[str, Any]:
    """FitVerdict's schema cut down to the fields the gate needs."""
    full = FitVerdict.model_json_schema()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: full["properties"][name] for name in _TRIAGE_FIELDS},
        "required": list(_TRIAGE_FIELDS),
    }
    if "$defs" in full:
        schema["$defs"] = full["$defs"]
    return schema


#: Enough for the three gate fields as JSON (about 20 tokens) with headroom.
TRIAGE_MAX_TOKENS = 64


#: FitVerdict cut down to the gate fields. The fit_score bounds are copied
#: by hand; tests/test_judges.py pins them to FitVerdict's.
class _TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_score: Annotated[int, Field(ge=0, le=100)]
    verdict: Verdict
    confidence: Confidence


def _read_tags(payload: Any) -> tuple[set[str], dict[str, str]]:
    """(the model names, each name's digest prefix) from `/api/tags`.

    An unexpected shape is a reason to report, never to raise: the checks
    that read this exist to prevent silent failure, so they must not become
    a new way for the scan to fail totally.
    """
    names: set[str] = set()
    digests: dict[str, str] = {}
    entries = payload.get("models") if isinstance(payload, dict) else None
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model") or ""
        if isinstance(name, str) and name:
            names.add(name)
            digest = entry.get("digest")
            if isinstance(digest, str):
                # Some servers print `sha256:<hex>`: the prefix names the
                # algorithm, not the weights, so it is not part of the 12.
                digests[name] = digest.removeprefix("sha256:")[:12]
    return names, digests


async def served_model_digest(cfg: LLMConfig) -> str | None:
    """The served model's digest prefix, read again from Ollama (2.5.8).

    For a run whose preflight could not name the weights (its 3 s probe timed
    out). Three answers, which the caller must not blur: a digest; "" when
    the server answered and lists the model with no digest, so it does not
    report one and "" is its honest identity; None when it could not be read
    (no answer, not JSON, the model not listed), so the identity is UNKNOWN
    and nothing may be keyed on it. Only Ollama reports a digest here: any
    other backend gets "".
    """
    if cfg.backend != "ollama" or not cfg.model:
        return ""
    url = f"{cfg.base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=IDENTITY_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("could not read the served model's digest from %s: %s", url, e)
        return None
    names, digests = _read_tags(payload)
    qualified = cfg.model if ":" in cfg.model else f"{cfg.model}:latest"
    if not names & {cfg.model, qualified}:
        return None
    return digests.get(cfg.model) or digests.get(qualified, "")


@register
class OllamaJudge(Judge):
    """A model running locally under Ollama. No key, no network egress, no bill.

    Ollama's /api/chat takes a JSON Schema in `format` and constrains sampling
    to it, so FitVerdict still arrives validated rather than coaxed out of
    prose — the same contract the Claude backend gives, locally.

    Measured against a live server, not only a mock: a 25-posting benchmark
    plus one 34-posting scan. Verdict accuracy 72% against hand-labelled
    expectations, and 0% score/verdict violations — the band table in `SYSTEM`
    and the verdict it returned never disagreed.

    Model-level blocker recall was 50%: the model itself named half the
    structural bars in the benchmark set. That is the number to design around
    rather than quote selectively. rolescan does not depend on it — configured
    `hard_blockers` are matched deterministically before the model is called
    and force `blocked` whatever it says, and `min_report_score` gates the
    rest — but a bar that appears only in the posting text, and that no
    configured term names, is one this backend will miss half the time.

    Quality is materially below Claude for this task; the point is that the
    tool works for someone who has no key and does not want one.
    """

    name = "ollama"
    needs_api_key = False
    description = (
        "Local model via Ollama. Free, offline, no key. Run against a live "
        "server: 72% verdict accuracy over 25 postings, 50% blocker recall."
    )

    cheap_triage = True

    #: The token counts of the last answer, or None before the first call or
    #: when the server reported none (2.5.8). One attribute per judge, shared
    #: by every call in flight (`llm.max_concurrent`), so with more than one it
    #: holds whichever answer landed last; and a call that fails does not
    #: reset it, so after an error it still holds the previous answer's counts.
    #: A diagnostic for a single call at a time, not a per-posting record.
    last_usage: OllamaUsage | None = None

    async def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        """POST, and once more after `RETRY_DELAY` on a timeout or a 5xx.

        Only those two: a refused connection is a server that is not running
        (the preflight's business, and a retry only doubles the wait), and a
        4xx will not change on a second asking.
        """
        try:
            return await self._post_once(url, payload)
        except httpx.TimeoutException:
            pass
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
        log.info("ollama call failed; retrying once in %.0fs", RETRY_DELAY)
        await asyncio.sleep(RETRY_DELAY)
        return await self._post_once(url, payload)

    async def _post_once(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response

    async def _chat(self, system: str, user: str, schema: dict[str, Any]) -> str:
        """One /api/chat round trip under `schema`, returning the raw content.

        Shared by `verdict` and `triage` so the two passes cannot drift apart
        in how they talk to the server - only in what they ask it for. Sends
        `ollama_options` (so `num_ctx` always), retries once on a timeout or
        a 5xx, and raises `PromptTruncatedError` when the answer's token
        counts show the prompt was cut (2.5.8).
        """
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": ollama_options(self.cfg),
        }
        try:
            response = await self._post(url, payload)
        except httpx.ConnectError as e:
            msg = (
                f"could not reach ollama at {self.cfg.base_url} ({e}). Start it "
                "with `ollama serve`, or set llm.backend to anthropic. Ollama "
                "listens on port 11434 by default."
            )
            raise RuntimeError(msg) from e
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:200]
            msg = f"ollama returned HTTP {e.response.status_code}: {detail}"
            raise RuntimeError(msg) from e

        try:
            body = response.json()
        except ValueError as e:
            # Same defect class as the preflight one below, one layer down: a
            # 200 that is not JSON. Contained by FitScorer, so it costs one
            # posting rather than the scan, but the raw JSONDecodeError is
            # what the digest would print at the reader.
            msg = f"ollama returned a 200 that was not JSON: {e}"
            raise RuntimeError(msg) from e
        self.last_usage = ollama_usage(
            body,
            prompt_chars=len(system) + len(user),
            num_ctx=self.cfg.num_ctx,
            check=self.cfg.check_truncation,
            window=self.context_window,
        )
        # Both layers are checked, not just the outer one. A body of
        # {"message": "hello"} is JSON, is a dict, and raises AttributeError
        # on .get - which is exactly the raw exception in the digest that
        # guarding the outer layer exists to prevent. Anything that is not a
        # string falls through to the empty content below, where the caller
        # turns it into the message a reader can act on.
        message = body.get("message") if isinstance(body, dict) else None
        raw = message.get("content") if isinstance(message, dict) else None
        return raw if isinstance(raw, str) else ""

    async def verdict(self, system: str, user: str) -> FitVerdict:
        content = await self._chat(system, user, FitVerdict.model_json_schema())
        try:
            return _without_rule(FitVerdict.model_validate_json(content))
        except ValueError as e:
            # Schema-constrained sampling should make this unreachable, but a
            # small model on an old Ollama can still return prose.
            msg = f"ollama did not return a usable verdict: {e}"
            raise RuntimeError(msg) from e

    async def facts(self, system: str, user: str) -> PostingFacts:
        content = await self._chat(system, user, PostingFacts.model_json_schema())
        try:
            return PostingFacts.model_validate_json(content)
        except ValueError as e:
            # Same defect class as `verdict`: schema-constrained sampling
            # should make this unreachable, but a small model on an old
            # Ollama can still return prose.
            msg = f"ollama did not return usable posting facts: {e}"
            raise RuntimeError(msg) from e

    async def triage(self, system: str, user: str) -> FitVerdict:
        """Score and route without paying for prose the digest will not print.

        Measured on 25 real postings: 18 decode tokens against 137, and 1.38s
        against 7.45s. The obvious worry is that `reason` acts as reasoning
        scaffolding, so that removing it moves the score and the gate routes on
        a different number than it reports. Tested over two independent
        25-posting slices, one long-description and one short: identical score
        AND verdict in all 50 cases, and no posting crossed the threshold.
        """
        content = await self._chat(system, user, triage_schema())
        try:
            fields = json.loads(content)
        except ValueError as e:
            msg = f"ollama did not return a usable triage verdict: {e}"
            raise RuntimeError(msg) from e
        if not isinstance(fields, dict):
            msg = (
                f"ollama returned a triage body that was not an object: {content[:120]}"
            )
            raise RuntimeError(msg)
        # Named explicitly rather than splatted: a server that answers with the
        # full schema (or any extra key) would otherwise pass `reason` twice
        # and raise TypeError from a call that actually succeeded.
        try:
            return FitVerdict(
                fit_score=fields["fit_score"],
                verdict=fields["verdict"],
                confidence=fields["confidence"],
                reason=TRIAGE_REASON,
            )
        except (KeyError, TypeError, ValidationError) as e:
            msg = f"ollama did not return a usable triage verdict: {e}"
            raise RuntimeError(msg) from e

    async def preflight(self) -> str:
        """A cheap reachability probe, run once before the scan starts.

        Deliberately not a chat completion: `/api/tags` is what `ollama list`
        calls, answered from memory with no model load, so it is safe to run
        on every scan without adding real latency. It also happens to be the
        one endpoint that can answer the second question that matters here —
        not just "is a server listening" but "does it have the model this
        config asks for" — which is the same class of silent failure and, in
        practice, the more common one: the server is up, the model was never
        pulled.
        """
        url = f"{self.cfg.base_url.rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=PREFLIGHT_TIMEOUT) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as e:
            # Caught BEFORE HTTPError, which it subclasses. A server that
            # answers with a status is running: telling its owner to `ollama
            # serve` sends them to start a process that is already up.
            return (
                f"ollama is running at {self.cfg.base_url} but /api/tags "
                f"returned HTTP {e.response.status_code}: "
                f"{e.response.text[:120]}. The server is up, so starting it "
                "again will not help - check the ollama log, or whether "
                "something else is bound to that port."
            )
        except httpx.HTTPError as e:
            # Nothing answered: refused, timed out, DNS, TLS.
            return (
                f"could not reach ollama at {self.cfg.base_url} ({e}). Start "
                "it with `ollama serve`, or set llm.backend to anthropic. "
                "Ollama listens on port 11434 by default."
            )
        except ValueError as e:
            # A 200 that is not JSON: a proxy, a captive portal, or another
            # service on 11434. json.JSONDecodeError is a ValueError, NOT an
            # httpx.HTTPError, so without this it escapes preflight entirely
            # and kills the scan before a single source is fetched.
            return (
                f"{url} answered, but not with JSON ({e}). Something other "
                "than ollama is probably listening on that port - check "
                "llm.base_url."
            )

        names, digests = _read_tags(payload)

        wanted = self.cfg.model
        # An untagged name means `:latest` to ollama, so `llama3.1` is NOT
        # satisfied by a server holding only `llama3.1:70b`: `ollama run
        # llama3.1` would try to pull. Comparing against bare prefixes said it
        # was, and the scan then failed one posting at a time.
        qualified = wanted if ":" in wanted else f"{wanted}:latest"
        if wanted and not names & {wanted, qualified}:
            available = ", ".join(sorted(names)) or "none"
            return (
                f"ollama is reachable at {self.cfg.base_url} but model "
                f"{wanted!r} is not pulled (available: {available}). Run "
                f"`ollama pull {wanted}`, or set llm.model to one that is."
            )
        if not wanted:
            return ""
        self.model_digest = digests.get(wanted) or digests.get(qualified, "")
        return await self._context_reason()

    async def _context_reason(self) -> str:
        """Why the model cannot hold `llm.num_ctx` tokens, or "" (2.5.8).

        `/api/show` reports the window the model was built for, as
        `<architecture>.context_length` in `model_info` (32,768 for
        qwen2.5:14b), kept on `context_window`. Ollama never runs a model past
        it, so a larger `num_ctx` is a window the server will not provide, and
        a prompt longer than the real one is cut with token counts that
        cannot show it: a scan does not score at all on this reason. A server
        that cannot answer (an older Ollama, a proxy) is not checked: this is
        a guard, not a new way for the scan to fail.
        """
        url = f"{self.cfg.base_url.rstrip('/')}/api/show"
        try:
            async with httpx.AsyncClient(timeout=PREFLIGHT_TIMEOUT) as client:
                response = await client.post(url, json={"model": self.cfg.model})
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("could not read the model's context window from %s: %s", url, e)
            return ""
        info = body.get("model_info") if isinstance(body, dict) else None
        window = _model_window(info) if isinstance(info, dict) else None
        self.context_window = window
        if window is None or window >= self.cfg.num_ctx:
            return ""
        return (
            f"the model's context window ({window}) is below llm.num_ctx "
            f"({self.cfg.num_ctx}); lower llm.num_ctx or pick a larger model"
        )


load_plugins()
