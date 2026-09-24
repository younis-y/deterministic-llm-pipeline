"""Configuration: a validated pydantic tree loaded from YAML.

Every knob lives here so behaviour changes are config edits, not code edits.
Secrets resolve from the environment when left blank, so the file itself stays
safe to commit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

__all__ = ["Config", "LLMConfig", "OutputConfig", "ProfileConfig", "SourceEntry"]


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
    location_penalty: int = 25
    min_keyword_score: Annotated[int, Field(ge=0)] = 18
    """Prefilter gate. Postings below this never reach the LLM, which is the
    single biggest lever on cost."""
    min_report_score: Annotated[int, Field(ge=0, le=100)] = 55
    """Final gate. Postings below this never reach the digest."""
    hard_blocker_score: Annotated[int, Field(ge=0)] = 50
    """`blockers` is a severity gradient, not a flat list of bans: a weight of
    60 next to "security clearance" and a weight of 15 next to "matlab" are
    both legitimate, but they mean different things. This is the line between
    them. A term weighted at or above it is a structural bar - the candidate
    cannot apply their way past it, so it forces `blocked` and the LLM does
    not get a vote. A term below it is a preference: it still subtracts its
    weight from the keyword score and still shapes ranking, but it must never
    force `blocked` on its own."""


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

        if not self.api_key:
            object.__setattr__(self, "api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        judge = available_judges().get(self.backend)
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
