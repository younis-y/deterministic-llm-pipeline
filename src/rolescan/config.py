"""Configuration: a validated pydantic tree loaded from YAML.

Every knob lives here so behaviour changes are config edits, not code edits.
Secrets resolve from the environment when left blank, so the file itself stays
safe to commit.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    model_validator,
)

from rolescan.models import JobField, Level, normalise_term

__all__ = [
    "Config",
    "LLMConfig",
    "OutputConfig",
    "ProfileConfig",
    "RulesConfig",
    "SourceEntry",
]

log = logging.getLogger(__name__)


class RulesConfig(BaseModel):
    """Rules for filtering and evaluating job postings per candidate.

    Rules live in config, not in the prompt, because multi-part rules are
    unreliable when left to the model's inference. Code applies them the same
    way every time. Each user's rules differ, so they belong here, not in a
    one-size-fits-all system prompt."""

    model_config = ConfigDict(extra="forbid")

    max_years_required: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Skip postings that state a minimum experience above this. None "
            "disables the check. A posting that states no years is never "
            "skipped by this rule."
        ),
    )
    allowed_levels: list[Level] = Field(
        default=[Level.graduate_entry, Level.junior, Level.mid, Level.not_stated],
        description=(
            "Career levels that pass the filter. `not_stated` is in the default "
            "because an advert that does not state a level must not be skipped "
            "for failing to state it."
        ),
    )
    student_only: Literal["skip", "allow"] = Field(
        default="allow",
        description=(
            "Skip or allow roles marked for students. Allow by default because "
            "the library cannot know whether the user is a student; a graduate "
            "sets this to 'skip'."
        ),
    )
    allowed_fields: list[JobField] | None = Field(
        default=None,
        description="Job fields that pass the filter. None means any field.",
    )
    max_graduation_year: int | None = Field(
        default=None,
        description=(
            "A stated graduation year above this skips; None = no limit. The "
            'year is the EARLIEST one the advert accepts, so "graduating 2027 '
            'or 2028" passes a limit of 2027. While this is set, a posting '
            "that states a graduation year is judged on the year alone and "
            "`student_only` does not fire for it; `student_only` still decides "
            "postings that state no year. Unset, a stated year changes nothing."
        ),
    )
    level_from_title_only: bool = Field(
        default=False,
        description=(
            "When true, only a level word in the job title can fire the "
            "level rule; a level the model states is ignored, even when its "
            "quote is copied from the title."
        ),
    )
    field_exempt_companies: list[str] = Field(
        default_factory=list,
        description=(
            "Employers whose postings pass the field rule whatever their field "
            "(matched on the company name, ignoring case, width and accents, "
            "as a whole word run). For the few firms where any entry role is "
            "wanted."
        ),
    )


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
    title_only_blockers: list[str] = Field(default_factory=list)
    """`blockers` terms that count only when they appear in the TITLE.

    For words that name a role's level or sector in a title but occur as
    boilerplate in bodies: `head of` ("head office"), `director` ("board of
    directors"), `military` ("military or veteran status", the US
    equal-opportunity line on every internship from a US employer). Measured
    2026-10-07: 82 cached adverts were rejected on weight alone, mostly these
    three. A term here keeps its weight from `blockers`. Only the weight is
    title-only: a term also in `hard_blockers` still bars on the whole text."""
    hidden_gate_margin: Annotated[int, Field(ge=0)] = 10
    """How far under `min_keyword_score` a reject may be and still be listed
    in "Hidden by your rules". 0 lists none. The gate is the stage that hides
    the most good roles (2026-10-07: 20 of 50 labelled good roles sat under
    it), and a reject is recorded as seen, so the listing is the only chance
    to notice a weight that is wrong."""
    location_penalty: int = 25
    min_keyword_score: Annotated[int, Field(ge=0)] = 18
    """Prefilter gate. Postings below this never reach the LLM, which is the
    single biggest lever on cost."""
    excluded_locations: list[str] = Field(default_factory=list)
    """Places the candidate cannot or will not work, matched against the
    posting's LOCATION and nothing else.

    A hard bar, so a match hides the role the way a clearance requirement does.
    Location-only on measured grounds: putting "united states" in `blockers`,
    which match the whole posting, would have deleted 133 real London and Dubai
    roles whose descriptions merely mention a US parent or office, and blocked
    zero actually-US ones. A global employer names its headquarters in every
    advert it writes."""
    nationalities: list[str] = Field(default_factory=list)
    """Nationalities the candidate holds, as the words an advert would use:
    the demonym and the country ("freedonian", "freedonia"). A nationality bar
    whose quote names one of them is not a bar for this candidate. Residence
    permits are not nationalities: a golden visa does not meet "UAE
    nationals only", so it is not listed here. A matching term kept in
    `hard_blockers` (for example "freedonian nationals only") still blocks at
    the keyword stage, so remove such terms from `hard_blockers` when listing
    the nationality."""
    agencies: list[str] = Field(default_factory=list)
    """Company names that post roles they are not themselves hiring for:
    recruiters, staffing firms, job boards. Matched against the COMPANY name
    only, never the description, so a posting that merely mentions a recruiter
    is unaffected."""
    agency_penalty: Annotated[int, Field(ge=0)] = 0
    """What an agency listing costs in keyword score. 0 disables it.

    A penalty rather than a bar, deliberately: plenty of good work is found
    through an agency, and the same role is often posted by both the employer
    and its recruiter. Ranking them below a direct listing lets the better
    version win a contested digest while an agency-only role still appears."""
    max_age_days: Annotated[int, Field(ge=0)] = 90
    """Drop postings older than this, whatever the source. 0 disables it.

    LinkedIn takes an age parameter in the search itself, so this never bit
    while it was the only source returning rows. ATS boards do not: Greenhouse
    and Ashby serve whatever is on the board, evergreen requisitions included,
    and a real digest carried a Jane Street posting dated 2024-02-15. A
    posting with no date at all is kept - unknown is not the same as old."""
    min_report_score: Annotated[int, Field(ge=0, le=100)] = 55
    """Final gate. Postings below this never reach the digest."""
    rules: RulesConfig | None = Field(
        default=None,
        description=(
            "Rules for filtering postings. None means no rule fires and "
            "verdicts come from fit score alone."
        ),
    )

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

        `title_only_blockers` goes through the same function so that it can be
        compared with the `blockers` keys it names.

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

        # An entry that normalises to nothing is dropped, as in `blockers`: it
        # cannot name a key, so it changes nothing.
        title_only = (normalise_term(raw) for raw in self.title_only_blockers)
        object.__setattr__(
            self, "title_only_blockers", list(dict.fromkeys(t for t in title_only if t))
        )

        # Normalise nationalities: strip, casefold, drop empties (2.5.7).
        object.__setattr__(
            self,
            "nationalities",
            [n.strip().casefold() for n in self.nationalities if n.strip()],
        )

        self._warn_inert_settings()
        return self

    def _warn_inert_settings(self) -> None:
        """Warn about two settings that load fine and then do nothing, or not
        what was meant. Warnings, not errors: neither can delete a role, and
        either may be deliberate while the config is being edited."""
        for term in self.title_only_blockers:
            if term not in self.blockers:
                log.warning(
                    "profile.title_only_blockers: %r has no weight in "
                    "profile.blockers, so it does nothing. Give it a weight in "
                    "blockers (it then counts in the job title only).",
                    term,
                )
        for term in self.hard_blockers:
            for nationality in self.nationalities:
                if re.search(rf"(?<!\w){re.escape(nationality)}(?!\w)", term):
                    log.warning(
                        "profile.hard_blockers: %r names %r, a nationality you "
                        "hold in profile.nationalities, but a hard_blockers "
                        "term blocks at the keyword stage regardless of "
                        "nationalities. Remove it to let the nationality rule "
                        "decide.",
                        term,
                        nationality,
                    )
                    break


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
    cascade: bool = True
    """Score in two passes: a cheap one that only settles the score, then the
    full verdict only for postings that clear `min_report_score`. The digest
    never prints reason or blockers for anything below that line,
    and generating them is ~85% of a local call. Measured at roughly half the
    scoring time on a real 531-posting run. Ignored by backends whose triage
    is not actually cheaper - see `Judge.cheap_triage`."""
    extra_prompt: str = ""
    """Appended to the system prompt verbatim, for context this library has no
    business knowing about.

    The seam exists so a private caller can add its own vocabulary - a set of
    documents to choose between, a house style, a client's constraints -
    without teaching the public prompt what any of it means. Empty by default,
    and an empty value changes the prompt not at all."""
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    """Sampling temperature. Zero by default: this is a classification task
    with a fixed rubric, and every benchmark this project has run measured
    accuracy at zero. Leaving it unset meant the local backend inherited
    Ollama's default of 0.8, so the same posting could score differently on
    a re-run and the measured accuracy never described what shipped.

    Both backends honour it (the hosted one sends it through the SDK's
    `extra_body`, 2.5.9). A model that rejects it, as `claude-sonnet-5` and
    `claude-sonnet-5-5` do for any value but the API's default of 1.0, is
    logged once and then called without it: its sampling is the API's."""
    description_chars: Annotated[int, Field(ge=500)] = 6000
    num_ctx: Annotated[int, Field(ge=2048)] = 12288
    """Context window, in tokens, asked of a local (Ollama) model on every
    call (2.5.8). Unset, Ollama picks its own from the machine's memory:
    32,768 on a 36 GB Mac, 4,096 on a smaller one, where it silently cut the
    middle (the instructions and worked examples) out of a 7,200-8,300 token
    facts prompt and still returned valid JSON. 12,288 holds the longest
    measured prompt (8,333 tokens) plus a full 1,500-token answer."""
    check_truncation: bool = True
    """Treat an Ollama answer whose token counts show a cut prompt (or a full
    window) as an error (2.5.8). The check assumes `prompt_eval_count` counts
    the whole prompt even when Ollama reuses its cache, as measured on the
    server this was built against. If a server or version counts only the
    uncached part, every call would look cut: set this to false, which keeps
    recording the counts but never judges them."""
    cache_days: Annotated[int, Field(ge=0)] = 30
    mode: Literal["facts", "judge"] = Field(
        default="facts",
        description=(
            "Scoring mode. facts: the model extracts quoted facts and code "
            "applies profile.rules. judge: one call, the model decides the "
            "verdict itself and profile.rules do not apply. Both are supported."
        ),
    )
    enricher: str = Field(
        default="",
        description=(
            "Name of a registered enricher that runs extra work only for "
            "postings that clear min_report_score. Empty means none."
        ),
    )
    facts_examples_file: Path | None = Field(
        default=None,
        description=(
            "A YAML list of worked examples (title, company, description, "
            "facts) rendered into the facts-mode SYSTEM prompt after its "
            "instructions, so they sit inside the cached prefix. Resolved "
            "relative to the config file and validated when the config loads. "
            "None means no examples and an unchanged prompt."
        ),
    )

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


#: Where a site owner can read what is calling (2.5.8): this project's page.
DEFAULT_CONTACT_URL = "https://github.com/younis-y/deterministic-llm-pipeline"


def default_user_agent(contact_url: str = DEFAULT_CONTACT_URL) -> str:
    """`rolescan/<version> (+<contact_url>)`, the User-Agent every request
    sends unless `http.user_agent` is set (2.5.8).

    2.5.7 still sent "rolescan/2.2 (personal job search tool)": three
    releases stale, and no way for a site owner to find out what was
    calling. The version is imported here, when a config is built, and not
    at the top of the module: `rolescan/__init__.py` imports this module
    before it sets `__version__`.
    """
    from rolescan import __version__

    return f"rolescan/{__version__} (+{contact_url})"


class HTTPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: float = 20.0
    max_concurrent: Annotated[int, Field(ge=1, le=64)] = 8
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    contact_url: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = DEFAULT_CONTACT_URL
    """Sent in the User-Agent, so a site owner can see what is calling and
    where to read about it (2.5.8). Point it at your fork, or a page of your
    own. A blank one is a config error, not `rolescan/<version> (+)`."""
    user_agent: str = ""
    """The User-Agent header. Empty, the default, means
    `rolescan/<version> (+<contact_url>)`; anything else is sent as it is."""

    @model_validator(mode="after")
    def _default_user_agent(self) -> Self:
        if not self.user_agent:
            object.__setattr__(self, "user_agent", default_user_agent(self.contact_url))
        return self


class EmailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    to: str = ""
    bind_interface: str = ""
    """The network interface whose IPv4 address the mail socket binds to, for
    example `en0`, for a machine where a VPN blocks the mail ports while HTTPS
    passes. The address is read at every send, so a changed network still works.
    Empty, the default, leaves the choice to the default route. Port 465 uses
    implicit TLS; any other port starts in plain text and upgrades with STARTTLS."""

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


class RetentionConfig(BaseModel):
    """How many days each cache keeps a row (2.5.8); 0 keeps rows for ever.

    Only caches: `seen` (the record of what has been listed) and
    `applications` (what the user did) are never trimmed. Before 2.5.8 only
    `verdicts` could be trimmed, by hand, and `postings` was 83% of the
    21 MB store after two weeks of daily scans."""

    model_config = ConfigDict(extra="forbid")

    postings: Annotated[int, Field(ge=0)] = 90
    """Cached posting pages. A trimmed page costs one re-fetch if its source
    still lists it. Never trimmed: a url with an `applications` row, or an
    apply/consider role still listed within this many days."""
    deferred: Annotated[int, Field(ge=0)] = 45
    """Deferral counts of postings not sighted for this long."""
    verdicts: Annotated[int, Field(ge=0)] = 180
    """Cached LLM facts and verdicts; `rolescan prune --days` overrides it.
    Never less than `llm.cache_days` (a verdict younger than that is still a
    valid cache hit), and with `llm.cache_days: 0` none is trimmed."""


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("digests")
    db_path: Path = Path("seen.db")
    max_roles: Annotated[int, Field(ge=1)] = 15
    show_blocked: bool = True
    """Blocked roles are still worth seeing once, so you know the market moved."""
    hidden_max: Annotated[int, Field(ge=0)] = 60
    """Most lines in the digest's "Hidden by your rules" section (2.6.0),
    counted across every rule, highest scores first. The rest are written to a
    file beside the digest, which names it. The section used to have no cap,
    at about a third of a kilobyte a line, so a long list pushed the alarms
    past the point where Gmail clips a message. 0 lists none: the counts and
    the file only. In the email the roles are laid out first and this section
    gets what they leave, so a high value never pushes a role out of it."""
    thin_unread_after: Annotated[int, Field(ge=1)] = 3
    """After this many runs without a description, a posting is listed once
    under "Unread" and recorded as seen. A posting with no text cannot be
    judged, so it is held back and looked at again next run; a source that
    never sends text (a Workday board with `details: false`, a structured page
    with no body) would otherwise hold it back for ever, and a pile of such
    postings would crowd out the ones that can be judged."""
    retention_days: RetentionConfig = Field(default_factory=RetentionConfig)
    backup_keep: Annotated[int, Field(ge=0)] = 7
    """Daily copies of the store kept in `backups/` beside it (2.5.8).
    `rolescan scan` takes the day's copy before it opens the store, so the
    first scan of a new version is covered before any migration runs. 0: no
    automatic copy; `rolescan backup` still copies on demand and never
    deletes."""
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
        examples = cfg.llm.facts_examples_file
        if examples is not None:
            # Resolved here, once, because FitScorer only ever sees `cfg.llm`
            # and never the config's location. Loaded here too, so a bad
            # example fails the run before a single posting is scored rather
            # than as one error per posting mid-scan.
            from rolescan.scoring.examples import load_facts_examples

            resolved = cfg.resolve(examples)
            object.__setattr__(cfg.llm, "facts_examples_file", resolved)
            load_facts_examples(resolved)
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
