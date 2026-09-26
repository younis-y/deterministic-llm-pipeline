"""Configuration: a validated pydantic tree loaded from YAML.

Every knob lives here so behaviour changes are config edits, not code edits.
Secrets resolve from the environment when left blank, so the file itself stays
safe to commit.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from rolescan.models import normalise_term

__all__ = ["Config", "LLMConfig", "OutputConfig", "ProfileConfig", "SourceEntry"]

log = logging.getLogger(__name__)


class SourceEntry(BaseModel):
    """One configured board. `kind` selects the registered source plugin."""

    model_config = ConfigDict(extra="allow")

    kind: str
    slug: str
    label: str = ""
    enabled: bool = True
    verified: bool = False

    @model_validator(mode="after")
    def _default_label(self) -> Self:
        if not self.label:
            object.__setattr__(self, "label", self.slug)
        return self

    @property
    def options(self) -> dict[str, Any]:
        """Extra source-specific keys, e.g. Workday's site and host.

        pydantic v2 keeps `extra="allow"` fields in `model_extra`, separate
        from the declared ones, so this reads there rather than `__dict__`.
        """
        return dict(self.model_extra or {})


class ProfileConfig(BaseModel):
    """Who the candidate is, and what counts as a good role for them."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    summary: str = Field(
        default="",
        description="Free text handed to the LLM as the candidate background.",
    )
    cv_dir: Path | None = Field(
        default=None,
        description="Directory of CV variant files, used for tailoring advice.",
    )
    locations: list[str] = Field(default_factory=list)
    allow_remote: bool = True
    keywords: dict[str, int] = Field(default_factory=dict)
    blockers: dict[str, int] = Field(default_factory=dict)
    """A severity gradient: points off the keyword score when the term is in
    the posting text. A weight says how much a term costs, and nothing else.
    Whether a term is survivable is `hard_blockers`."""
    hard_blockers: list[str] = Field(default_factory=list)
    """Structural bars: things no application can get past. A term here forces
    a `blocked` verdict that no LLM opinion can override.

    Separate from `blockers` because the two answer different questions on
    different timescales. A weight is retuned whenever the prefilter is
    calibrated; whether a clearance or a passport is a wall is a fact about
    the candidate that changes once a decade. Folding both into one number
    made two reasonable configurations unexpressible: a fatal bar that should
    only cost 10 points, and a 60-point preference that must not block.

    A term listed here keeps whatever weight `blockers` gives it, and costs
    nothing if `blockers` does not mention it. Entries are normalised at load
    and matched on word boundaries - see `rolescan.scoring.keyword`."""
    location_penalty: int = 25
    min_keyword_score: Annotated[int, Field(ge=0)] = 18
    """Prefilter gate. Postings below this never reach the LLM, which is the
    single biggest lever on cost."""
    min_report_score: Annotated[int, Field(ge=0, le=100)] = 55
    """Final gate. Postings below this never reach the digest."""

    @model_validator(mode="after")
    def _clean_blocker_terms(self) -> Self:
        """Normalise every blocker term at load, and refuse the dangerous ones.

        Both fields are normalised here, with the same function, because both
        are matched the same way: a search over `Job.blob`, which is
        casefolded with its whitespace collapsed. A key the loader leaves
        alone is a term that can never match - `security  clearance` with two
        spaces, or a trailing space picked up from a YAML quote, is a weight
        that is never charged and nothing says so. Normalising only one of
        them was worse than normalising neither: the same term could be hard
        (normalised, matching) and weightless (unnormalised, never matching)
        at once, with the two fields disagreeing silently.

        `hard_blockers` is additionally checked for two mistakes worth failing
        the load over, because that list can delete a role from the digest
        permanently: the posting is dropped AND written to `seen`, so the
        reader cannot recover from it.

        Normalisation can also collapse two `blockers` keys onto one term,
        which then has to resolve to a single weight. The heaviest wins, and
        the collapse is logged. Heaviest rather than last-written because
        "last" depends on the order two lines happen to sit in the YAML: that
        is invisible to the author, and it would change the prefilter when the
        file is merely re-sorted. A weight is also a cost, so taking the larger
        of the two can never quietly weaken a bar the author did write.

        An entry that normalises to nothing matches every posting, so every
        role would be blocked and, with `output.show_blocked` false, silently
        deleted - the one configuration that means "everything is hard".

        A one-character entry is the same defect with a smaller blast radius:
        boundary matching makes it a bar on a single standalone letter, which
        most prose contains. Neither is worth guessing the author's intent
        over. Misspellings cannot be caught here, which is why the cheap
        checks that CAN be made are made loudly.
        """
        weighted: dict[str, int] = {}
        for raw_key, weight in self.blockers.items():
            # An empty key is dropped rather than refused: unlike a hard bar
            # it cannot delete anything, it only shifted every posting by a
            # constant (`"" in blob` is always true), and a term that is not
            # a term has no weight to charge.
            key = normalise_term(raw_key)
            if not key:
                continue
            if key in weighted and weighted[key] != weight:
                log.warning(
                    "profile.blockers: %r and an earlier key both normalise to "
                    "%r. Keeping the heavier weight %d and dropping %d.",
                    raw_key,
                    key,
                    max(weighted[key], weight),
                    min(weighted[key], weight),
                )
            weighted[key] = max(weighted.get(key, weight), weight)
        object.__setattr__(self, "blockers", weighted)

        cleaned: list[str] = []
        for i, raw in enumerate(self.hard_blockers):
            term = normalise_term(raw)
            if not term:
                msg = (
                    f"profile.hard_blockers[{i}] ({raw!r}) is empty. An empty "
                    "hard blocker matches every posting, so every role would "
                    "be blocked - and with output.show_blocked false, deleted "
                    "from the digest without a trace."
                )
                raise ValueError(msg)
            if len(term) < 2:
                msg = (
                    f"profile.hard_blockers[{i}] ({raw!r}) is a single "
                    "character. Hard blockers are matched on word boundaries, "
                    "so this bars any posting containing that letter on its "
                    "own. Write the whole phrase."
                )
                raise ValueError(msg)
            if term not in cleaned:
                cleaned.append(term)
        object.__setattr__(self, "hard_blockers", cleaned)
        return self


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: str = "anthropic"
    """Which LLM plugin scores postings. `anthropic` needs a key; `ollama`
    runs a local model and needs none. See `rolescan backends`."""
    base_url: str = "http://localhost:11434"
    """Only read by local backends. Ollama's default listener."""
    timeout: float = 120.0
    """Local models are slow. This is per posting, not per run."""
    model: str = "claude-sonnet-5"
    """Sonnet is the right tier here: the task is judgement over a short
    document, run tens of times a day."""
    api_key: str = ""
    max_tokens: int = 1500
    max_concurrent: Annotated[int, Field(ge=1, le=32)] = 5
    max_calls_per_run: Annotated[int, Field(ge=0)] = 60
    """Hard ceiling. Stops a badly tuned prefilter turning into a big bill."""
    description_chars: Annotated[int, Field(ge=500)] = 6000
    cache_days: Annotated[int, Field(ge=0)] = 30

    _auto_disabled: bool = PrivateAttr(default=False)
    """Set when `_resolve_key` switched scoring off, rather than the user."""

    @property
    def wants_scoring(self) -> bool:
        """Whether the config asks for LLM scoring at all.

        Not the same question as `enabled`, because this model answers that
        one itself: a hosted backend with no key is switched off here, which
        is precisely the case that most needs reporting at startup. Someone
        who wrote `llm.enabled: false` has made a choice and does not need
        telling their backend is unusable; someone whose key vanished does.
        """
        return self.enabled or self._auto_disabled

    @model_validator(mode="after")
    def _resolve_key(self) -> Self:
        """Disable hosted scoring when there is no key — but only hosted.

        Disabling unconditionally was correct while Claude was the only
        backend and wrong the moment a local one existed: a model on
        localhost has no key and never will, so an empty api_key would have
        silently switched the scorer off for exactly the users who chose the
        backend to avoid needing one.
        """
        from rolescan.scoring.judges import available_judges

        judge = available_judges().get(self.backend)
        if not self.api_key:
            # The judge names its own variable, so a third-party backend
            # declaring api_key_env = "OPENAI_API_KEY" is read from the one it
            # asked for. Reading ANTHROPIC_API_KEY unconditionally made this a
            # public plugin API that names a variable it never looks at: the
            # user exports it, the config still self-disables, and the message
            # tells them to export what they just exported.
            env = (
                judge.api_key_env
                if judge and judge.api_key_env
                else "ANTHROPIC_API_KEY"
            )
            object.__setattr__(self, "api_key", os.environ.get(env, ""))
        needs_key = judge.needs_api_key if judge else True
        if self.enabled and needs_key and not self.api_key:
            object.__setattr__(self, "enabled", False)
            self._auto_disabled = True
        return self


class HTTPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: float = 20.0
    max_concurrent: Annotated[int, Field(ge=1, le=64)] = 8
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    user_agent: str = "rolescan/2.2 (personal job search tool)"


class EmailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    to: str = ""

    @model_validator(mode="after")
    def _resolve(self) -> Self:
        if not self.password:
            # JOBSCAN_SMTP_PASS is the pre-rename name. It is the only name this
            # project ever exported outside its own tree, so an existing shell
            # profile or crontab still carries it; without the fallback the
            # digest would silently stop being emailed. Deprecated, not removed.
            password = os.environ.get("ROLESCAN_SMTP_PASS") or os.environ.get(
                "JOBSCAN_SMTP_PASS", ""
            )
            object.__setattr__(self, "password", password)
        if self.enabled and not (self.smtp_host and self.password):
            object.__setattr__(self, "enabled", False)
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("digests")
    db_path: Path = Path("seen.db")
    max_roles: Annotated[int, Field(ge=1)] = 15
    show_blocked: bool = True
    """Blocked roles are still worth seeing once, so you know the market moved."""
    email: EmailConfig = Field(default_factory=EmailConfig)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    sources: list[SourceEntry] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    root: Path = Field(default=Path(), exclude=True)

    @classmethod
    def load(cls, path: Path) -> Config:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls.model_validate(data)
        object.__setattr__(cfg, "root", path.resolve().parent)
        return cfg

    def resolve(self, p: Path) -> Path:
        """Interpret relative paths against the config file, not the cwd.

        Without this, a cron job with a different working directory silently
        writes its database somewhere new and re-reports every role as fresh.
        """
        return p if p.is_absolute() else self.root / p

    @property
    def enabled_sources(self) -> list[SourceEntry]:
        return [s for s in self.sources if s.enabled]
