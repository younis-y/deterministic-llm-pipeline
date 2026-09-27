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

import importlib.util
import logging
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Any, ClassVar

import httpx

from rolescan.config import LLMConfig
from rolescan.models import FitVerdict

__all__ = [
    "Judge",
    "available_judges",
    "get_judge",
    "register",
    "unusable_backend_reason",
]

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Judge]] = {}

#: How long a preflight liveness check is allowed to take. Short on purpose:
#: this runs before every scan, not just when something is already wrong, so
#: it must never become a new way for the tool to hang. `LLMConfig.timeout`
#: is for a chat completion and is far too long for this.
PREFLIGHT_TIMEOUT = 3.0


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

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    async def verdict(self, system: str, user: str) -> FitVerdict:
        """One posting, one judgement. Raise on failure; FitScorer counts it."""

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


async def unusable_backend_reason(cfg: LLMConfig) -> str:
    """Why the configured judge cannot score anything, or "" if it can.

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
        return ""
    cls = _REGISTRY.get(cfg.backend)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        return (
            f"llm.backend {cfg.backend!r} is not a registered backend "
            f"(registered: {known})"
        )
    if cls.needs_api_key and not cfg.api_key:
        env = cls.api_key_env or "the appropriate environment variable"
        return (
            f"backend {cfg.backend!r} is configured but no API key is visible "
            f"(export {env}, or set llm.api_key in the config)"
        )
    if cls.requires_module and not _importable(cls.requires_module):
        return (
            f"backend {cfg.backend!r} needs the {cls.requires_module} package, "
            f"which is not installed (pip install 'rolescan[{cfg.backend}]')"
        )
    try:
        return await cls(cfg).preflight()
    except Exception as e:
        # A judge is a plugin and `cls(cfg)` now runs at startup, so a broken
        # __init__ or a preflight that raises instead of returning would take
        # the whole scan down before a single source was fetched: no digest,
        # no email, a traceback in a launchd log nobody reads. Treated as
        # "this backend is unusable", which is what it is.
        log.warning("preflight for backend %s raised: %s", cfg.backend, e)
        return (
            f"backend {cfg.backend!r} could not be checked: "
            f"{type(e).__name__}: {e}"
        )


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

    async def verdict(self, system: str, user: str) -> FitVerdict:
        response = await self._get_client().messages.parse(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=FitVerdict,
        )
        parsed = response.parsed_output
        if not isinstance(parsed, FitVerdict):  # pragma: no cover - enforced upstream
            msg = f"model returned {type(parsed).__name__}, expected FitVerdict"
            raise RuntimeError(msg)
        return parsed


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

    async def verdict(self, system: str, user: str) -> FitVerdict:
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "format": FitVerdict.model_json_schema(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_predict": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
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
        # Both layers are checked, not just the outer one. A body of
        # {"message": "hello"} is JSON, is a dict, and raises AttributeError
        # on .get - which is exactly the raw exception in the digest that
        # guarding the outer layer exists to prevent. Anything that is not a
        # string falls through to the empty content below, where
        # model_validate_json turns it into the message a reader can act on.
        message = body.get("message") if isinstance(body, dict) else None
        raw = message.get("content") if isinstance(message, dict) else None
        content = raw if isinstance(raw, str) else ""
        try:
            return FitVerdict.model_validate_json(content)
        except ValueError as e:
            # Schema-constrained sampling should make this unreachable, but a
            # small model on an old Ollama can still return prose.
            msg = f"ollama did not return a usable verdict: {e}"
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

        names: set[str] = set()
        entries = payload.get("models") if isinstance(payload, dict) else None
        for entry in entries or []:
            # An unexpected shape is a reason to report, never to raise: this
            # check exists to prevent silent failure, so it must not become a
            # new way for the scan to fail totally.
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model") or ""
            if isinstance(name, str) and name:
                names.add(name)

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
        return ""


load_plugins()
