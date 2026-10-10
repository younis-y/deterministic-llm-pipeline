"""Write docs/config.md from the pydantic config models.

    python scripts/gen_config_doc.py           write docs/config.md
    python scripts/gen_config_doc.py --check   exit 1 when the file is stale

Keys, types, defaults and enum values come from the models, so they cannot
drift. The one-line descriptions are written by hand in DESCRIPTIONS below,
because a model's docstring is a design note and not user documentation. A
field with no entry stops the script: adding a config key means saying what it
does, here, in the same change. `tests/test_config_doc.py` fails when the
committed file differs from what this prints.
"""

from __future__ import annotations

import enum
import sys
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from rolescan.config import (
    Config,
    EmailConfig,
    HTTPConfig,
    LLMConfig,
    OutputConfig,
    ProfileConfig,
    RetentionConfig,
    RulesConfig,
    SourceEntry,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "config.md"


class MissingDescription(KeyError):  # noqa: N818 - a script failure, not an API error
    """A config field has no hand-written description in DESCRIPTIONS."""


#: Defaults that are computed or read from the environment, so the model's own
#: default would mislead.
DEFAULT_TEXT: dict[str, str] = {
    "llm.api_key": "from the environment",
    "http.contact_url": "the project's page",
    "http.user_agent": "`rolescan/<version> (+<contact_url>)`",
    "output.email.password": "from `ROLESCAN_SMTP_PASS`",
}

DESCRIPTIONS: dict[str, str] = {
    # profile
    "profile.name": "A label for yourself. Nothing reads it.",
    "profile.summary": (
        "Your background in your own words. Sent as it is to a model backend as "
        "the candidate; keyword-only scoring never reads it. Numbers beat "
        "adjectives."
    ),
    "profile.locations": (
        "Places you will work. A posting whose location contains none of these "
        "(a case-insensitive substring) is charged `location_penalty`. A posting "
        "with no stated location is not penalised. Empty turns the check off."
    ),
    "profile.allow_remote": "A remote posting is never charged `location_penalty`.",
    "profile.keywords": (
        "Weighted terms. Each is matched as a plain, case-insensitive SUBSTRING "
        "of the posting text, not a whole word, so a short term matches inside "
        "longer ones: `ai` hits `maintain` and `email`. Write `machine learning` "
        "or `ai engineer`, not `ai`. A term found in the title counts three "
        "times its weight; a term counts once however often it occurs."
    ),
    "profile.blockers": (
        "Terms that cost points. Each is a case-insensitive WHOLE-WORD match "
        "(`director` does not hit `directorate`) and subtracts its weight from "
        "the keyword score. A posting pushed under `min_keyword_score` is "
        "dropped before scoring and recorded as seen. A preference, not a bar: "
        "see `hard_blockers`."
    ),
    "profile.hard_blockers": (
        "Structural bars: things no application can get past, such as a "
        "clearance or a nationality gate. A whole-word, case-insensitive match "
        "marks the posting `blocked` whatever a model says. Keep it short and "
        "specific: with `output.show_blocked: false` a blocked posting is "
        "removed and never shown again. An empty or one-character entry is a "
        "config error."
    ),
    "profile.title_only_blockers": (
        "`blockers` terms whose weight counts only when the term is in the job "
        "title, for words such as `head of` that are boilerplate in a body "
        "(`head office`). A term here needs a weight in `blockers`. One also in "
        "`hard_blockers` still bars on the whole text."
    ),
    "profile.hidden_gate_margin": (
        "How far under `min_keyword_score` a posting may fall and still be "
        'listed under "Hidden by your rules" in the digest, so a weight that '
        "rejects good roles shows up. `0` lists none."
    ),
    "profile.location_penalty": "Points subtracted when the location check fails.",
    "profile.min_keyword_score": (
        "The prefilter gate. A posting scoring below this never reaches a "
        "model, which makes it the biggest lever on cost."
    ),
    "profile.excluded_locations": (
        "Places you cannot or will not work. A whole-word, case-insensitive "
        "match against the posting's LOCATION only, never its description. A "
        "match blocks the posting like a `hard_blockers` term."
    ),
    "profile.nationalities": (
        "Nationalities you hold, in the words an advert uses: the demonym and "
        "the country. A nationality bar whose quote names one is not a bar for "
        "you. Acts in facts mode only. A matching term in `hard_blockers` still "
        "blocks, so remove it from there."
    ),
    "profile.agencies": (
        "Company names that post roles they are not hiring for: recruiters, "
        "staffing firms. A whole-word match against the COMPANY name only. Does "
        "nothing unless `agency_penalty` is above 0."
    ),
    "profile.agency_penalty": (
        "Points an agency listing loses. `0` turns it off. A penalty and not a "
        "bar, so an agency-only role still appears while the employer's own "
        "posting ranks above it."
    ),
    "profile.max_age_days": (
        "Drop postings older than this many days, from sources whose dates mean "
        "freshness (aggregators). Employer boards are exempt, and so is a "
        "posting with no date. `0` turns it off."
    ),
    "profile.min_report_score": (
        "The final gate: a posting scoring below this is not in the digest. "
        "Model scores and keyword-only scores are on different scales, and a "
        "keyword-only run wants a lower number (about 20 to 30)."
    ),
    "profile.rules": (
        "Eligibility rules, applied by code in facts mode to the facts a model "
        "quotes from each posting. Omit them and no rule fires. See "
        "`profile.rules` below."
    ),
    # profile.rules
    "profile.rules.max_years_required": (
        "Skip a posting that states a minimum above this many years. A posting "
        "that states no years is never skipped by it."
    ),
    "profile.rules.allowed_levels": (
        "Levels that pass the level rule. Keep `not_stated`, or every advert "
        "that states no level is skipped."
    ),
    "profile.rules.student_only": (
        "`skip` drops roles open only to current students; `allow` keeps them."
    ),
    "profile.rules.allowed_fields": (
        "Fields that pass the field rule; unset means any field. The field is "
        "read from the title first."
    ),
    "profile.rules.max_graduation_year": (
        "Skip a stated graduation year above this, the earliest year the advert "
        "accepts. Unset is no limit. While set, `student_only` applies only to "
        "adverts that state no year."
    ),
    "profile.rules.level_from_title_only": (
        "`true`: only a level word in the job title can fire the level rule, "
        "and a level the model states is ignored."
    ),
    "profile.rules.field_exempt_companies": (
        "Employers whose postings pass the field rule whatever their field: a "
        "company name, matched as whole words, ignoring case and accents."
    ),
    # llm
    "llm.enabled": (
        "`false` scores on keywords alone: no key, no network, no cost. Also "
        "switched off, with a note in the digest, when a hosted backend has no "
        "key."
    ),
    "llm.backend": (
        "Which plugin scores postings: `anthropic` (needs `ANTHROPIC_API_KEY`) "
        "or `ollama` (a local model, no key). `rolescan backends` lists what is "
        "registered."
    ),
    "llm.base_url": "Where a local backend listens. Ollama's default listener.",
    "llm.timeout": "Seconds per posting for a local model, which can be slow.",
    "llm.model": (
        "The model the backend is asked for. For `ollama`, one you have pulled."
    ),
    "llm.api_key": (
        "The key for a hosted backend. Leave it blank to read the backend's "
        "environment variable (`ANTHROPIC_API_KEY`). If you write a key here, "
        "keep this file out of version control."
    ),
    "llm.max_tokens": "The longest answer asked of the model, in tokens.",
    "llm.max_concurrent": "Model calls in flight at once.",
    "llm.max_calls_per_run": (
        "Hard ceiling on model calls in one scan. Postings past it are deferred "
        "to the next run, not dropped. `0` allows no calls."
    ),
    "llm.max_minutes": (
        "Wall-clock budget for the model in one scan, in minutes, counted from "
        "the first model call. Once it is spent no new call starts, calls "
        "already running finish, and the postings not yet scored are deferred "
        "to the next run, as the call ceiling defers them. `0` is no time "
        "limit."
    ),
    "llm.cascade": (
        "Score in two passes: a short call that settles the score, then the "
        "full verdict only for postings that clear `min_report_score`. Used "
        "only by a backend whose short call is cheaper."
    ),
    "llm.extra_prompt": (
        "Text appended to the judge-mode prompt. Not sent in facts mode, where "
        "only an enricher may use it."
    ),
    "llm.temperature": (
        "Sampling temperature. `0` suits classification against a fixed rubric. "
        "A model that refuses the setting is called without it from then on, "
        "and the log says so once."
    ),
    "llm.description_chars": (
        "How many characters of a posting's description are sent to a model."
    ),
    "llm.num_ctx": (
        "Context window, in tokens, asked of a local (Ollama) model on every "
        "call. rolescan checks before the scan that the model can hold it, and "
        "refuses a model whose own window is smaller."
    ),
    "llm.check_truncation": (
        "Treat an answer whose token counts show a cut prompt as an error. Set "
        "`false` only for an Ollama that counts just the uncached part of a "
        "prompt."
    ),
    "llm.cache_days": "Days a cached model answer stays valid. `0` never expires.",
    "llm.mode": (
        "`facts` (default): the model extracts quoted facts and code applies "
        "`profile.rules`. `judge`: one call per posting, the model gives the "
        "score, verdict and blockers itself, `profile.rules` do not apply and "
        '`llm.extra_prompt` is used. A supported mode: see the README, "Judge '
        'mode".'
    ),
    "llm.enricher": (
        "Name of a registered enricher (entry-point group `rolescan.enrichers`) "
        "that does extra work only for postings clearing `min_report_score`. "
        "Empty means none. A failing enricher keeps the plain verdict."
    ),
    "llm.facts_examples_file": (
        "A YAML list of worked examples (title, company, description, facts) "
        "added to the facts-mode prompt. Relative to this file, and validated "
        "when the config loads. Unset means no examples."
    ),
    # http
    "http.timeout": "Seconds allowed per request.",
    "http.max_concurrent": "Requests in flight at once.",
    "http.max_retries": "Retries per failed request.",
    "http.contact_url": (
        "Shown in the User-Agent, `rolescan/<version> (+<contact_url>)`, so a "
        "site owner can see what is calling. Point it at your fork or a page of "
        "your own. A blank value is a config error."
    ),
    "http.user_agent": "Replaces the whole User-Agent header when set.",
    "http.bind_interface": (
        "The network interface whose IPv4 address the job-fetching connections "
        "leave from (`en0`, `eth0`), for a machine where a VPN's exit address is "
        "refused by a job site that answers the machine's own. Read once when a "
        "scan or `discover` starts; an interface with no IPv4 address stops the "
        "run before any request. The model backends are never bound. A "
        "destination reachable over IPv6 only cannot be reached from an IPv4 "
        "source address, and `HTTP_PROXY` and `HTTPS_PROXY` are not used while "
        "it is set. Empty leaves it to the default route."
    ),
    # output
    "output.dir": "Where digests are written, relative to this file.",
    "output.db_path": "The store (SQLite), relative to this file.",
    "output.max_roles": (
        "Most roles in one digest. The rest are deferred to the next run, not dropped."
    ),
    "output.show_blocked": (
        "List blocked roles at the bottom of the digest. `false` removes them, "
        "and they are recorded as seen."
    ),
    "output.hidden_max": (
        'Most lines in the digest\'s "Hidden by your rules" section, counted '
        "across every rule, highest scores first; each rule's count is shown "
        "either way. The rest are written to a `-hidden.md` file beside the "
        "digest, which names it. In the email the roles get the room first and "
        "this section gets what they leave (a line costs about 0.3 KB, and "
        "Gmail clips a message past about 102 KB), so a high value never "
        "pushes a role out; the email lists as many as fit, and the Markdown "
        "digest lists up to this many. `0` lists none."
    ),
    "output.thin_unread_after": (
        "Runs a posting may arrive without text before it is listed once under "
        '"Unread" and recorded.'
    ),
    "output.reopen_programme_days": (
        "Days a programme posting must go unlisted before it counts as new "
        'again. A posting whose title names a programme ("Summer Internship", '
        '"Graduate Programme", an off-cycle role, an insight week) and that '
        "no scan has found on a board for this long is scored and listed "
        "again, with a line saying how long it was off the board. A posting a "
        "board lists on every scan never comes back, and a title that is not a "
        "programme is unchanged. `0` turns it off."
    ),
    "output.retention_days": (
        "How many days each cache keeps a row. See `output.retention_days` below."
    ),
    "output.backup_keep": (
        "Daily copies of the store kept in `backups/` beside it. `0` turns the "
        "automatic copy off; `rolescan backup` still copies on demand."
    ),
    "output.email": "Sending the digest by email. See `output.email` below.",
    # output.retention_days
    "output.retention_days.postings": (
        "Cached posting pages. A trimmed page costs one re-fetch. `0` keeps them all."
    ),
    "output.retention_days.deferred": (
        "Deferral counts of postings not sighted for this long. `0` keeps them all."
    ),
    "output.retention_days.verdicts": (
        "Cached model answers. Never less than `llm.cache_days`. `0` keeps them all."
    ),
    # output.email
    "output.email.enabled": (
        "Send the digest by email. Switched off, with a note, when `smtp_host` "
        "or the password is missing."
    ),
    "output.email.smtp_host": "The mail server.",
    "output.email.smtp_port": (
        "`587` connects in plain text and upgrades with STARTTLS; `465` uses "
        "implicit TLS, encrypted from the first byte. Either way the server's "
        "certificate must verify against your system's trusted authorities; a "
        "relay with a self-signed certificate is refused."
    ),
    "output.email.username": "The login name on the mail server.",
    "output.email.password": (
        "The mail password. Leave it blank and `ROLESCAN_SMTP_PASS` is read."
    ),
    "output.email.to": "Where the digest is sent.",
    "output.email.bind_interface": (
        "The network interface whose IPv4 address the mail socket binds to "
        "(`en0`, `eth0`), for a machine where a VPN blocks the mail ports while "
        "HTTPS passes. Looked up at every send. Empty leaves it to the default "
        "route."
    ),
    # sources
    "sources[].kind": (
        "A registered source kind. `rolescan sources` lists them with their "
        "slug formats."
    ),
    "sources[].slug": (
        "The employer's board identifier; its format depends on the kind."
    ),
    "sources[].label": "The name shown in the digest. Defaults to the slug.",
    "sources[].enabled": (
        "`false` keeps the entry in the file but skips it; `discover` marks it "
        "disabled."
    ),
    "sources[].verified": (
        "A note to yourself that `discover` confirmed the entry. Nothing reads it."
    ),
}

#: Options a source reads from its own entry (kept by `SourceEntry`'s
#: `extra="allow"`). Not in a pydantic model, so listed by hand; a test fails
#: when a source reads an option this table does not name.
SOURCE_OPTIONS: dict[str, list[tuple[str, str, str]]] = {
    "workday": [
        ("site", "the slug", "The site name in the careers URL."),
        ("host", "`wd3`", "The Workday host number in the careers URL."),
        (
            "details",
            "`true`",
            "Fetch each posting's description (one extra request each). "
            "`false` lists titles only.",
        ),
        (
            "max_rows",
            "`500`",
            "Postings read per listing pass. A board holding more is reported "
            "as cut short.",
        ),
        (
            "applied_facets",
            "none",
            "Workday's own filters: facet parameter to a list of value ids, "
            "taken from the board's listing answer.",
        ),
        (
            "search_text",
            "none",
            "One listing pass per text (a string or a list); each posting is "
            "kept once.",
        ),
    ],
    "structured": [
        ("sitemap", "none", "The sitemap URL. Required."),
        (
            "url_pattern",
            "`/job`",
            "A regex; only sitemap URLs it matches are fetched.",
        ),
        (
            "exclude_pattern",
            "none",
            "A regex; matching URLs are dropped before any fetch, for a board "
            "that mixes regions.",
        ),
        (
            "delay",
            "`0.5`",
            "Seconds between detail-page requests. A `Crawl-delay` in the "
            "host's robots.txt raises it to that (never past 30 seconds).",
        ),
        (
            "max_pages",
            "`500`",
            "Most detail pages read per run. A sitemap with more in scope is "
            "reported as cut short.",
        ),
        (
            "max_age_days",
            "unset",
            "Skip a URL whose sitemap `lastmod` is more than this many days "
            "old, before any fetch. A URL with no readable `lastmod` is kept.",
        ),
        (
            "incremental",
            "`false`",
            "`true` reads only the URLs whose `lastmod` is newer than the last "
            "scan that read its whole window, newest first, up to `max_pages`; "
            "a scan with no mark reads everything in scope up to the same cap. "
            "The mark moves only after a real scan that was recorded with "
            "nothing deferred, no model failure, no read cut short at "
            "`max_pages` and no page that failed for now. Otherwise the next "
            "scan opens the same window again, so a cut read repeats until "
            "`max_pages` or `max_age_days` covers the window. It compares the "
            "sitemap's `lastmod` and nothing else, so a URL first listed with "
            "a `lastmod` older than the mark (a posting the board published "
            "late, or under an old date) is not read; set it `false` for one "
            "scan to sweep the whole window. Its counts vary by design, so "
            "the quiet and shrink alarms skip it. A `--dry` run keeps no "
            "mark. To read everything once more, set it `false` for a scan.",
        ),
        (
            "max_sitemap_urls",
            "`50000`",
            "Refuse a sitemap that lists more URLs than this, with an error "
            "that says how to narrow the read.",
        ),
    ],
    "adzuna": [
        (
            "app_id",
            "from `ADZUNA_APP_ID`",
            "Your Adzuna application id. Leave it blank to read the environment.",
        ),
        (
            "app_key",
            "from `ADZUNA_APP_KEY`",
            "Your Adzuna application key. Leave it blank to read the environment.",
        ),
        ("queries", "none", "A list of search texts, one search each. Required."),
        ("max_days_old", "`7`", "Only postings no older than this."),
        ("where", "none", "A place to search near."),
        (
            "max_pages",
            "`5`",
            "Pages read per query. A query with more results than "
            "`max_pages` x `results_per_page` is reported as cut short (with the "
            "API's own count when the entry has one query).",
        ),
        (
            "results_per_page",
            "`50`",
            "Rows a page asks for, 1 to 50 (Adzuna's own limit). A page with "
            "fewer rows is the last one.",
        ),
    ],
    "reed": [
        (
            "api_key",
            "from `REED_API_KEY`",
            "Your Reed API key. Leave it blank to read the environment. With "
            "neither, the source is skipped, and the digest says so.",
        ),
        ("queries", "none", "A list of search texts, one search each. Required."),
        ("where", "none", "A place to search near (Reed's `locationName`)."),
        ("distance", "unset", "Miles around `where`. Reed's own default applies."),
        ("graduate", "`false`", "`true` asks for graduate roles only."),
        (
            "direct_employer_only",
            "`false`",
            "`true` leaves out roles posted by a recruitment agency.",
        ),
        (
            "max_pages",
            "`5`",
            "Pages read per query. A query with more results than "
            "`max_pages` x `results_per_page` is reported as cut short (with "
            "Reed's own count when the entry has one query).",
        ),
        (
            "results_per_page",
            "`100`",
            "Rows a page asks for, 1 to 100 (Reed's own limit). A page with "
            "fewer rows is the last one.",
        ),
    ],
    "jooble": [
        (
            "api_key",
            "from `JOOBLE_API_KEY`",
            "Your Jooble API key. Leave it blank to read the environment. "
            "With neither, the source is skipped, and the digest says so.",
        ),
        ("queries", "none", "A list of search texts, one search each. Required."),
        ("where", "none", "A place to search (Jooble's `location`)."),
        ("radius", "unset", "Kilometres around `where`."),
        ("salary", "unset", "The least pay wanted, as Jooble reads it."),
        (
            "max_pages",
            "`5`",
            "Pages read per query. Jooble sets the page size, so a query is "
            "read until a page comes back empty, its own count is reached, or "
            "this many pages are read; a query cut at this is reported as cut "
            "short.",
        ),
    ],
    "workable_search": [
        ("queries", "none", "A list of search texts, one search each. Required."),
        ("where", "none", "A place to search (Workable's `location`)."),
        (
            "max_pages",
            "`5`",
            "Pages read per query. Workable sets the page size, so a query is "
            "read until no further page is offered, its own count is reached, "
            "or this many pages are read; a query cut at this is reported as "
            "cut short.",
        ),
        (
            "delay",
            "`2`",
            "Seconds between requests. Queries run one after another, and "
            "pages within a query too, a `delay` apart, never together. `0` "
            "turns the wait off. A read also stops, and says why in the "
            "digest, when a page repeats rows already read or Workable gives "
            "back a page token it was already sent, which means the paging "
            "parameter is not being honoured.",
        ),
    ],
}

#: (title, intro) per documented model, in output order. The key is the
#: dotted prefix its fields carry in DESCRIPTIONS.
SECTIONS: list[tuple[str, type[BaseModel], str]] = [
    ("profile", ProfileConfig, "Who you are and what counts as a good role."),
    (
        "profile.rules",
        RulesConfig,
        "Eligibility rules for facts mode. Each fires only on a fact the advert "
        "states, and every skip names the rule and quotes the advert.",
    ),
    ("llm", LLMConfig, "How postings are scored, and by what."),
    ("http", HTTPConfig, "How requests are made."),
    ("output", OutputConfig, "Where results go, and how long caches live."),
    (
        "output.retention_days",
        RetentionConfig,
        "Days a cache keeps a row. Never touches what has been seen, or what "
        "you marked.",
    ),
    ("output.email", EmailConfig, "Emailing the digest."),
]


def _walk(ann: Any) -> list[Any]:
    """Every type inside an annotation, outermost first."""
    found = [ann]
    for arg in get_args(ann):
        found.extend(_walk(arg))
    return found


def _enum_in(ann: Any) -> type[enum.Enum] | None:
    for inner in _walk(ann):
        if isinstance(inner, type) and issubclass(inner, enum.Enum):
            return inner
    return None


def _split_optional(ann: Any) -> tuple[Any, bool]:
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        rest = [a for a in get_args(ann) if a is not type(None)]
        if len(rest) == 1:
            return rest[0], True
    return ann, False


def _type_label(field: FieldInfo) -> str:
    ann, _optional = _split_optional(field.annotation)
    origin = get_origin(ann)
    if origin is Literal:
        label = " or ".join(f'`"{v}"`' for v in get_args(ann))
    elif origin is list:
        (item,) = get_args(ann)
        kind = _scalar(item)
        label = "list of choices" if kind == "choice" else f"list of {kind}"
    elif origin is dict:
        key, value = get_args(ann)
        label = f"map of {_scalar(key)} to {_scalar(value)}"
    elif isinstance(ann, type) and issubclass(ann, BaseModel):
        label = "section"
    else:
        label = _scalar(ann)
    bounds = _bounds(field)
    return f"{label}, {bounds}" if bounds else label


def _scalar(ann: Any) -> str:
    ann, _ = _split_optional(ann)
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return "choice"
    return {
        str: "text",
        int: "whole number",
        float: "number",
        bool: "true or false",
        Path: "path",
    }.get(ann, getattr(ann, "__name__", str(ann)))


def _bounds(field: FieldInfo) -> str:
    low = high = None
    # pydantic keeps `Field(ge=..., le=...)` as metadata objects with `ge` and
    # `le` attributes; read the attributes rather than import annotated_types,
    # which is pydantic's dependency, not this project's.
    for meta in field.metadata:
        if (ge := getattr(meta, "ge", None)) is not None:
            low = ge
        if (le := getattr(meta, "le", None)) is not None:
            high = le
    if low is not None and high is not None:
        return f"{low} to {high}"
    if low is not None:
        return f"{low} or more"
    if high is not None:
        return f"{high} or less"
    return ""


def _plain(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _default_label(path: str, field: FieldInfo) -> str:
    if path in DEFAULT_TEXT:
        return DEFAULT_TEXT[path]
    ann, _ = _split_optional(field.annotation)
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return "see below"
    value = field.get_default(call_default_factory=True)
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, list | dict):
        if not value:
            return "empty"
        return ", ".join(f"`{_plain(v)}`" for v in value)
    if value == "":
        return "empty"
    return f"`{_plain(value)}`"


def _description(path: str, field: FieldInfo) -> str:
    if path not in DESCRIPTIONS:
        msg = f"{path} has no entry in DESCRIPTIONS (scripts/gen_config_doc.py)"
        raise MissingDescription(msg)
    text = DESCRIPTIONS[path]
    choices = _enum_in(field.annotation)
    if choices is not None:
        listed = ", ".join(f"`{member.value}`" for member in choices)
        text += f" Values: {listed}."
    return text


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _table(prefix: str, model: type[BaseModel]) -> list[str]:
    rows = ["| Key | Type | Default | What it does |", "|---|---|---|---|"]
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}"
        rows.append(
            f"| `{path}` | {_cell(_type_label(field))} "
            f"| {_cell(_default_label(path, field))} "
            f"| {_cell(_description(path, field))} |"
        )
    return rows


def _sources() -> list[str]:
    out = [
        "## sources",
        "",
        "A list. Each entry names one board. Beyond the keys below, an entry "
        "carries the options of its kind; any other key is kept and passed to "
        "the source, so a misspelt option is silently ignored: check it with "
        "`rolescan discover`.",
        "",
        "| Key | Type | Default | What it does |",
        "|---|---|---|---|",
    ]
    for name, field in SourceEntry.model_fields.items():
        path = f"sources[].{name}"
        default = "required" if field.is_required() else _default_label(path, field)
        out.append(
            f"| `{path}` | {_cell(_type_label(field))} | {default} "
            f"| {_cell(_description(path, field))} |"
        )
    out += ["", "### Options by kind", ""]
    out.append("| Kind | Option | Default | What it does |")
    out.append("|---|---|---|---|")
    for kind, options in SOURCE_OPTIONS.items():
        for option, default, text in options:
            out.append(f"| `{kind}` | `{option}` | {default} | {_cell(text)} |")
    out += [
        "",
        "`greenhouse`, `lever`, `ashby`, `workable` and `smartrecruiters` take "
        "only the keys above.",
        "",
    ]
    return out


HEADER = """\
# Configuration reference

Generated from the config models by `scripts/gen_config_doc.py`. Do not edit it
by hand: a test fails when it differs from what the script prints. Copy
`config.example.yaml` to `config.yaml` to start, or `examples/quickstart.yaml`
for the smallest working file.

Unknown keys are an error (the config says which), except the per-source
options under `sources`. Relative paths resolve against the config file, not
the working directory. Anything left blank that names a secret is read from the
environment.

## How terms are matched

Three matchers are in play, and they differ on purpose.

- `profile.keywords` are plain, case-insensitive **substrings**. `ai` matches
  `maintain`, `email` and `said`. Write the whole phrase.
- `profile.blockers`, `profile.hard_blockers`, `profile.title_only_blockers`,
  `profile.excluded_locations`, `profile.agencies` and
  `profile.rules.field_exempt_companies` match **whole words**, ignoring case:
  `director` does not match `directorate`. A blocker can delete a role, so it
  has to mean the word.
- `profile.locations` is a case-insensitive substring of the posting's location
  field.
"""


def render() -> str:
    """The whole of docs/config.md."""
    out = [HEADER]
    for prefix, model, intro in SECTIONS:
        level = "##" if "." not in prefix else "###"
        out += [f"{level} {prefix}", "", intro, "", *_table(prefix, model), ""]
    out += _sources()
    out += [
        "## Environment variables",
        "",
        "| Variable | Read when |",
        "|---|---|",
        "| `ANTHROPIC_API_KEY` | `llm.api_key` is blank and the backend is "
        "`anthropic` |",
        "| `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` | an `adzuna` entry has no "
        "`app_id` or `app_key` |",
        "| `REED_API_KEY` | a `reed` entry has no `api_key` |",
        "| `JOOBLE_API_KEY` | a `jooble` entry has no `api_key` |",
        "| `ROLESCAN_SMTP_PASS` | `output.email.password` is blank |",
        "",
        "rolescan does not read a `.env` file: export the variable in your shell "
        "or scheduler.",
        "",
    ]
    # Config's own top-level keys, so a new section cannot be forgotten.
    documented = {"profile", "llm", "http", "output", "sources"}
    missing = set(Config.model_fields) - documented - {"root"}
    if missing:
        msg = f"top-level config sections not documented: {sorted(missing)}"
        raise MissingDescription(msg)
    return "\n".join(out)


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != text:
            sys.stderr.write(
                f"{TARGET} is stale: run python scripts/gen_config_doc.py\n"
            )
            return 1
        return 0
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
