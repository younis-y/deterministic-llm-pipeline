"""Command line interface."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.table import Table

from rolescan.config import Config, SourceEntry
from rolescan.digest import render_html, render_markdown, send_email, write_digest
from rolescan.http import Fetcher
from rolescan.models import Verdict
from rolescan.pipeline import ScanResult, record_scan, run_scan
from rolescan.slugs import SlugIndex
from rolescan.sources import available, get_source
from rolescan.sources.base import ProbeResult, ProbeStatus
from rolescan.store import PruneReport, Store, StoreTooNewError, refuse_a_newer_store
from rolescan.storefile import BackupError, RunLockedError, backup, run_lock

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Typed, async job scanner with LLM fit scoring and CV matching.",
)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]

_VERDICT_STYLE = {
    Verdict.APPLY: "bold green",
    Verdict.CONSIDER: "yellow",
    Verdict.SKIP: "dim",
    Verdict.BLOCKED: "red",
}

_PROBE_STYLE = {
    ProbeStatus.OK: "bold green",
    ProbeStatus.EMPTY: "green",
    ProbeStatus.UNKNOWN: "bold yellow",
    ProbeStatus.SKIPPED: "dim",
    ProbeStatus.FAIL: "bold red",
}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _email_skip_reason(cfg: Config) -> str:
    """Why no email is going out, in the reader's terms.

    `EmailConfig` switches itself off when the host or the password is
    missing, so by the time we get here "disabled" covers three different
    situations and only one of them is deliberate. Printing the right one is
    the difference between a two-second fix and assuming the digest was sent.
    """
    email = cfg.output.email
    if not email.smtp_host:
        return "no output.email.smtp_host is set in the config"
    if not email.password:
        return (
            "no SMTP password is visible — export ROLESCAN_SMTP_PASS "
            "(~/.zshenv, so launchd's non-interactive shell sources it)"
        )
    return "output.email.enabled is false in the config"


def _load(path: Path) -> Config:
    if not path.is_file():
        console.print(f"[red]No config at {path}[/]. Copy config.example.yaml.")
        raise typer.Exit(2)
    try:
        return Config.load(path)
    except Exception as e:
        console.print(f"[red]Config error:[/] {e}")
        raise typer.Exit(2) from e


def _refuse_a_newer_store(db_path: Path) -> None:
    """Exit 1 with the message, before anything is copied or opened, when the
    store was written by a newer rolescan (2.5.8).

    Before the backup on purpose: the day's copy would rotate out the oldest,
    and after `backup_keep` such days the copy taken before the upgrade, the
    one the message points at, would be gone."""
    try:
        refuse_a_newer_store(db_path)
    except StoreTooNewError as e:
        console.print(
            f"[bold red]{e}[/]\n[dim]Nothing was changed, and no copy was taken.[/]"
        )
        raise typer.Exit(1) from e


def _warn_if_llm_did_not_run(result: ScanResult, cfg: Config) -> None:
    """The loudest thing this command prints, deliberately.

    Every silent self-disabling defect this tool has had ended here: a judge
    that could not run, a run that reported success, and an empty digest with
    nothing in it saying why.
    """
    if result.enricher_unusable:
        # Its own note, printed regardless of `llm_unusable`: the judge is
        # fine here, only the (optional) extra step an enricher adds is not,
        # so this must never look like the backend warnings below.
        console.print(
            f"\n[yellow]{result.enricher_unusable}.[/] "
            "[dim]Postings are scored normally, just without that extra "
            "step.[/]\n"
        )
    if result.facts_cache_skipped:
        console.print(
            "\n[yellow]Facts cache skipped: model identity unknown.[/] "
            "[dim]The model's digest could not be read, so cached facts were "
            "neither used nor written this run and every posting was sent to "
            "the model.[/]\n"
        )
    if not result.llm_unusable:
        return
    if not result.llm_scored:
        console.print(
            f"\n[bold red]LLM scoring did not run: {result.llm_unusable}.[/]\n"
            "[bold red]Every posting was ranked on keyword score alone, "
            f"against min_report_score={cfg.profile.min_report_score}, which "
            "is calibrated for LLM fit scores — so this digest is probably "
            "empty for that reason and not because the market is.[/]\n"
            "[dim]Fix the backend, or run with --no-llm to accept a "
            "keyword-only digest without this warning.[/]\n"
        )
        return
    # Scoring worked, so the claim above would be false and the reader would
    # be looking at fit scores under a line saying there are none. Still worth
    # printing: the probe is short by design, and one that keeps failing on a
    # healthy server is a real thing to know.
    console.print(
        f"\n[yellow]The pre-scan backend check failed "
        f"({result.llm_unusable}), but scoring ran anyway: "
        f"{result.llm_scored - result.llm_cached} scored, "
        f"{result.llm_cached} from cache.[/]\n"
        "[dim]The liveness probe is deliberately short so an unattended "
        "run cannot hang on it; a loaded server can exceed it.[/]\n"
    )


def _ranked_table(result: ScanResult) -> Table:
    table = Table(title="Ranked", show_edge=False, header_style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Verdict")
    table.add_column("Role")
    table.add_column("CV")
    for item in result.reportable:
        table.add_row(
            str(item.score),
            f"[{_VERDICT_STYLE[item.verdict]}]{item.verdict.value}[/]",
            f"{item.job.title} — {item.job.company}",
        )
    return table


def _deliver(
    text: str, html_body: str, cfg: Config, path: Path, *, email: bool
) -> None:
    """Send the digest, or say exactly why it is not being sent.

    Never falls through in silence. A disabled emailer used to look exactly
    like a successful send from the caller's side, which under launchd means
    nobody finds out until they wonder why the 06:30 mail stopped arriving.
    """
    if not email:
        console.print("[dim]email skipped: --no-email[/]")
        return
    if not cfg.output.email.enabled:
        console.print(f"[yellow]email skipped:[/] {_email_skip_reason(cfg)}")
        return
    try:
        if send_email(text, cfg.output.email, html_body=html_body):
            console.print("[green]emailed[/]")
    except Exception as e:
        # The digest is already on disk, so nothing is lost - but the run did
        # not do what it was scheduled to do, and launchd only records that if
        # the exit status says so.
        console.print(f"[red]email failed:[/] {e}")
        console.print(f"[dim]the digest is still on disk at {path}[/]")
        raise typer.Exit(1) from e


#: Shown under `rolescan scan --help`; README.md lists the same codes.
_SCAN_EXIT_STATUS = (
    "Exit status: 0 ok; 1 email failed, the day's backup failed its check, or "
    "the store was written by a newer rolescan (takes precedence over 3); "
    "2 config error; 3 LLM scoring failed as a whole (after the digest was "
    "written and sent); 4 another run holds the store's lock."
)


@app.command(epilog=_SCAN_EXIT_STATUS)
def scan(
    config: ConfigOpt = Path("config.yaml"),
    dry: Annotated[
        bool,
        typer.Option(
            "--dry",
            help="Do not mark anything seen; write digest-dry.md, not latest.md.",
        ),
    ] = False,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Keyword scoring only.")
    ] = False,
    email: Annotated[bool, typer.Option("--email/--no-email")] = True,
    verbose: VerboseOpt = False,
) -> None:
    """Fetch, score, and write a digest of new roles."""
    _setup_logging(verbose)
    cfg = _load(config)
    if no_llm:
        cfg.llm.enabled = False
    db_path = cfg.resolve(cfg.output.db_path)
    try:
        with run_lock(db_path):
            _scan(cfg, config, dry=dry, no_llm=no_llm, email=email)
    except RunLockedError as e:
        # Exit 4, not 0: the run did not happen, and a wrapper (launchd, a
        # scheduler) must be able to tell that from a quiet day.
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(4) from e


def _scan(cfg: Config, config: Path, *, dry: bool, no_llm: bool, email: bool) -> None:
    """`scan`'s body, run while this process holds the store's run lock."""
    db_path = cfg.resolve(cfg.output.db_path)
    _refuse_a_newer_store(db_path)
    if cfg.output.backup_keep:
        try:
            backup(db_path, keep=cfg.output.backup_keep)
        except BackupError as e:
            # The copy failed its integrity check or could not be written:
            # either way this is the moment to stop writing to the store, not
            # to carry on and rotate out the last good copy.
            console.print(
                f"[bold red]{e}[/]\n[dim]Nothing was scanned. Restore a copy "
                f"from {db_path.parent / 'backups'}, or free disk space, then "
                "run again.[/]"
            )
            raise typer.Exit(1) from e

    async def _go() -> tuple[ScanResult, list[tuple[str, str, str]]]:
        result = await run_scan(cfg, dry_run=dry, check_llm=not no_llm)
        async with Store(db_path) as store:
            shortlist_rows = await store.shortlist()
        return result, shortlist_rows

    result, shortlist_rows = asyncio.run(_go())
    _warn_if_llm_did_not_run(result, cfg)

    # Both parts of the email are rendered from the same ScanResult. The HTML
    # one is not made from `text`: doing that shipped markdown source as the
    # HTML alternative, so the apply links were not links.
    text = render_markdown(
        result, shortlist=shortlist_rows, config_path=config.resolve()
    )
    html_body = render_html(
        result, shortlist=shortlist_rows, config_path=config.resolve()
    )
    path = write_digest(
        text, cfg.resolve(cfg.output.dir), name="digest-dry.md" if dry else None
    )

    # Recorded only now that the digest is on disk (2.5.7): a crash or a
    # failed write before this point leaves `seen` untouched, so the next run
    # sees the same postings again instead of losing them. And recorded
    # straight away, before any console rendering: the digest exists, so an
    # exception while drawing it to the terminal must not leave its postings
    # unrecorded and reported again tomorrow.
    asyncio.run(record_scan(cfg, result))
    if not dry:
        try:
            asyncio.run(_prune(cfg))
        except Exception as e:
            # The digest is written and its postings recorded; a cache trim
            # that fails must not turn a delivered run into a failed one.
            console.print(f"[yellow]could not trim old caches: {e}[/]")

    console.print(Markdown(text))
    console.print(f"\n[dim]written to {path}[/]")

    if result.reportable:
        console.print(_ranked_table(result))

    _deliver(text, html_body, cfg, path, email=email)

    if failure := result.llm_failure:
        # After the digest is written, recorded and sent: the digest says
        # what happened, and the exit status says it to whatever ran us.
        console.print(f"[bold red]{failure}.[/] [dim]Exit status 3.[/]")
        raise typer.Exit(3)


def _probe_table(rows: list[tuple[SourceEntry, ProbeResult]]) -> Table:
    table = Table(title="Configured sources", show_edge=False, header_style="bold")
    for col in ("", "Kind", "Slug", "Label", "Jobs", "Note"):
        table.add_column(col)
    for entry, result in rows:
        style = _PROBE_STYLE[result.status]
        label = entry.label + ("" if entry.enabled else " [dim](disabled)[/]")
        table.add_row(
            f"[{style}]{result.status.value}[/]",
            entry.kind,
            entry.slug,
            label,
            str(result.count) if result.count >= 0 else "-",
            result.detail,
        )
    return table


def _probe_summary(rows: list[tuple[SourceEntry, ProbeResult]]) -> str:
    """The one line worth reading if you read nothing else.

    `trustworthy` rather than "not failed": a board that answered 200 with
    zero rows is not evidence the slug is right, and counting it as verified
    is the bug this command exists to avoid.
    """
    verified = sum(1 for _, r in rows if r.trustworthy)
    unknown = sum(1 for _, r in rows if r.status is ProbeStatus.UNKNOWN)
    skipped = sum(1 for _, r in rows if r.status is ProbeStatus.SKIPPED)
    failed = sum(1 for _, r in rows if r.status is ProbeStatus.FAIL)
    return (
        f"\n[green]{verified} verified[/], [yellow]{unknown} unverifiable[/], "
        f"[dim]{skipped} skipped[/], [red]{failed} broken[/]  "
        f"(of {len(rows)} configured)"
    )


def _slug_hints(rows: list[tuple[SourceEntry, ProbeResult]]) -> list[str]:
    """Where to find the real slug, for the kinds that actually failed."""
    kinds = {e.kind for e, r in rows if r.status is ProbeStatus.FAIL}
    return [
        f"  [cyan]{name:16}[/] {cls.slug_hint}"
        for name, cls in sorted(available().items())
        if name in kinds and cls.slug_hint
    ]


async def _probe_all(cfg: Config) -> list[tuple[SourceEntry, ProbeResult]]:
    async with Fetcher(cfg.http) as fetcher:

        async def one(entry: SourceEntry) -> tuple[SourceEntry, ProbeResult]:
            return entry, await get_source(entry, fetcher).probe()

        return list(await asyncio.gather(*(one(e) for e in cfg.sources)))


@app.command()
def discover(
    config: ConfigOpt = Path("config.yaml"),
    verbose: VerboseOpt = False,
) -> None:
    """Probe every configured board and report which slugs actually work."""
    _setup_logging(verbose)
    cfg = _load(config)
    rows = asyncio.run(_probe_all(cfg))
    statuses = {r.status for _, r in rows}

    console.print(_probe_table(rows))
    console.print(_probe_summary(rows))

    if ProbeStatus.UNKNOWN in statuses:
        console.print(
            "\n[yellow]UNKNOWN[/] means the API answered 200 with zero rows and "
            "does not 404 unknown slugs, so an empty board and a wrong slug are\n"
            "indistinguishable. Open the careers page in a browser to settle it."
        )
    if ProbeStatus.FAIL in statuses:
        console.print("\n[bold]Where the slug comes from:[/]")
        for line in _slug_hints(rows):
            console.print(line)


@app.command()
def slugs(
    query: Annotated[list[str], typer.Argument(help="Company name(s) to resolve.")],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Directory of harvested *_companies.json."),
    ] = Path("ats-data"),
    kind: Annotated[
        str | None, typer.Option("--kind", help="Restrict to one ATS.")
    ] = None,
    limit: Annotated[int, typer.Option(help="Candidates per company.")] = 5,
) -> None:
    """Resolve real board slugs from a harvested ATS company directory.

    Guessing slugs is why most of the shipped config fails. Point this at a
    downloaded copy of a public ATS company dataset and it will tell you what
    the real identifiers are, with paste-ready config lines.
    """
    index = SlugIndex.load(data)
    if not index:
        console.print(
            f"[red]No usable dataset in {data}[/].\n\n"
            "Download one first, for example the CC BY-NC 4.0 dataset at\n"
            "  [cyan]https://github.com/Feashliaa/job-board-aggregator[/] "
            "(data/*_companies.json)\n"
            "then re-run with [cyan]--data <that directory>[/]."
        )
        raise typer.Exit(1)

    summary = ", ".join(f"{k} {n:,}" for k, n in sorted(index.kinds.items()))
    console.print(f"[dim]{len(index):,} slugs indexed: {summary}[/]\n")

    paste: list[str] = []
    for term in query:
        hits = index.search(term, limit=limit, kind=kind)
        table = Table(title=term, show_edge=False, header_style="bold")
        for col in ("Score", "Kind", "Slug", "Name"):
            table.add_column(col)
        if not hits:
            console.print(f"[yellow]{term}: no match[/]")
            continue
        for c in hits:
            style = "bold green" if c.score >= 0.9 else "yellow"
            table.add_row(f"[{style}]{c.score:.2f}[/]", c.kind, c.slug, c.name or "-")
        console.print(table)
        paste.append(hits[0].config_line)

    if paste:
        console.print("\n[bold]Best guess per company, for config.yaml:[/]\n")
        for line in paste:
            console.print(f"[cyan]{line}[/]")
        console.print(
            "\n[dim]Verify with `rolescan discover` before trusting these. "
            "Workday entries need a real `site` value from the careers URL.[/]"
        )


@app.command()
def backends() -> None:
    """List the LLM scoring backends, and which of them need a credential.

    rolescan runs with no credentials at all: every source but Adzuna is a
    public endpoint and keyword scoring is pure Python. The backend below is
    only for the optional second stage.
    """
    from rolescan.scoring import available_judges

    table = Table(title="LLM backends", show_edge=False, header_style="bold")
    table.add_column("Name")
    table.add_column("Needs a key")
    table.add_column("What it is")
    for name, cls in sorted(available_judges().items()):
        needs = (
            f"[yellow]{cls.api_key_env or 'yes'}[/]"
            if cls.needs_api_key
            else "[green]no[/]"
        )
        table.add_row(name, needs, cls.description)
    console.print(table)
    console.print(
        "\nSet [cyan]llm.backend[/] in config.yaml, or [cyan]llm.enabled: false[/] "
        "to score on keywords alone."
    )


@app.command()
def sources() -> None:
    """List every registered source plugin, built-in and third-party."""
    table = Table(title="Registered sources", show_edge=False, header_style="bold")
    table.add_column("Kind")
    table.add_column("Class")
    table.add_column("Slug format")
    for name, cls in sorted(available().items()):
        table.add_row(name, cls.__name__, cls.slug_hint or "-")
    console.print(table)


@app.command()
def show(config: ConfigOpt = Path("config.yaml")) -> None:
    """Reprint the most recent digest."""
    cfg = _load(config)
    path = cfg.resolve(cfg.output.dir) / "latest.md"
    if not path.is_file():
        console.print("[yellow]No digest yet. Run `rolescan scan`.[/]")
        raise typer.Exit(1)
    console.print(Markdown(path.read_text(encoding="utf-8")))


@app.command()
def stats(config: ConfigOpt = Path("config.yaml")) -> None:
    """How many postings the store has seen."""
    cfg = _load(config)

    async def go() -> int:
        async with Store(cfg.resolve(cfg.output.db_path)) as store:
            return await store.count()

    console.print(f"{asyncio.run(go())} postings recorded.")


def _verdict_retention(cfg: Config) -> int:
    """Days a cached verdict is kept by an automatic or configured trim; 0
    keeps them all.

    Never less than `llm.cache_days`: a verdict younger than that is still a
    valid cache hit, and trimming it would make the next scan pay for the
    same answer again. So it is `max(retention_days.verdicts, cache_days)`,
    and `cache_days: 0` (a verdict never expires) means none is trimmed.
    `rolescan prune --days N` is an explicit request and bypasses this.
    """
    keep, cache = cfg.output.retention_days.verdicts, cfg.llm.cache_days
    return 0 if keep == 0 or cache == 0 else max(keep, cache)


async def _prune(cfg: Config, *, verdicts_days: int | None = None) -> PruneReport:
    """Trim the store's caches to `output.retention_days`."""
    keep = cfg.output.retention_days
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        return await store.prune_all(
            postings_days=keep.postings,
            deferred_days=keep.deferred,
            verdicts_days=(
                _verdict_retention(cfg) if verdicts_days is None else verdicts_days
            ),
        )


@app.command()
def prune(
    config: ConfigOpt = Path("config.yaml"),
    days: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Drop cached verdicts older than this, whatever llm.cache_days "
            "says (default: output.retention_days.verdicts, never less than "
            "llm.cache_days).",
        ),
    ] = None,
) -> None:
    """Trim old caches: verdicts, postings, deferral counts, source counts."""
    cfg = _load(config)
    try:
        with run_lock(cfg.resolve(cfg.output.db_path)):
            _refuse_a_newer_store(cfg.resolve(cfg.output.db_path))
            report = asyncio.run(_prune(cfg, verdicts_days=days))
    except RunLockedError as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(4) from e
    console.print(
        f"Removed {report.verdicts} cached verdicts, {report.postings} cached "
        f"postings, {report.deferred} deferral counts and {report.source_counts} "
        "old source counts" + ("; the file was compacted." if report.vacuumed else ".")
    )


@app.command("backup")
def backup_store(config: ConfigOpt = Path("config.yaml")) -> None:
    """Copy the store to backups/ beside it, checked with integrity_check.

    Keeps the newest output.backup_keep daily copies. With backup_keep: 0 the
    scan takes no copy of its own, and this command copies on demand and never
    deletes one."""
    cfg = _load(config)
    db_path = cfg.resolve(cfg.output.db_path)
    _refuse_a_newer_store(db_path)
    try:
        path = backup(db_path, keep=cfg.output.backup_keep, force=True)
    except BackupError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    if path is None:
        console.print(f"[yellow]No store at {db_path} yet: nothing to back up.[/]")
        return
    console.print(f"Backed up to {path} (integrity_check ok).")


@app.command()
def mark(
    url: Annotated[
        str, typer.Argument(help="The posting URL, as printed in the digest.")
    ],
    state: Annotated[str, typer.Argument(help="shortlist, applied or dismissed.")],
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Record what you did with a posting so the digest stops repeating it."""
    if state not in Store.STATES:
        console.print(f"[red]state must be one of: {', '.join(Store.STATES)}[/]")
        raise typer.Exit(2)
    cfg = _load(config)

    async def go() -> None:
        async with Store(cfg.resolve(cfg.output.db_path)) as store:
            # Fill in company and title from the posting cache when we have
            # it. Without this a `mark shortlist` writes a permanently blank
            # row, and the digest's Shortlist section can only show the url.
            cached = await store.get_posting(url)
            company = cached[1].company if cached else ""
            title = cached[1].title if cached else ""
            await store.mark(url, state, company=company, title=title)

    asyncio.run(go())
    console.print(f"{state}: {url}")


@app.command()
def unsee(
    key: Annotated[
        str,
        typer.Argument(help="The posting URL (as printed in the digest) or its uid."),
    ],
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Forget a posting so the next scan can report it again."""
    cfg = _load(config)

    async def go() -> int:
        async with Store(cfg.resolve(cfg.output.db_path)) as store:
            return await store.unsee(key)

    n = asyncio.run(go())
    console.print(f"{n} row{'s' if n != 1 else ''} removed for {key}")


if __name__ == "__main__":
    app()
