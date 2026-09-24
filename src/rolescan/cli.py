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
from rolescan.pipeline import ScanResult, run_scan
from rolescan.scoring import CVLibrary
from rolescan.slugs import SlugIndex
from rolescan.sources import available, get_source
from rolescan.sources.base import ProbeResult, ProbeStatus
from rolescan.store import Store

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


@app.command()
def scan(
    config: ConfigOpt = Path("config.yaml"),
    dry: Annotated[
        bool, typer.Option("--dry", help="Do not mark anything seen.")
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

    async def _go() -> tuple[ScanResult, list[tuple[str, str, str]]]:
        result = await run_scan(cfg, dry_run=dry, check_llm=not no_llm)
        async with Store(cfg.resolve(cfg.output.db_path)) as store:
            shortlist_rows = await store.shortlist()
        return result, shortlist_rows

    result, shortlist_rows = asyncio.run(_go())

    if result.llm_unusable:
        # The loudest thing this command prints, deliberately. Every silent
        # self-disabling defect this tool has had ended here: a judge that
        # could not run, a run that reported success, and an empty digest with
        # nothing in it saying why.
        console.print(
            f"\n[bold red]LLM scoring did not run: {result.llm_unusable}.[/]\n"
            "[bold red]Every posting was ranked on keyword score alone, "
            f"against min_report_score={cfg.profile.min_report_score}, which "
            "is calibrated for LLM fit scores — so this digest is probably "
            "empty for that reason and not because the market is.[/]\n"
            "[dim]Fix the backend, or run with --no-llm to accept a "
            "keyword-only digest without this warning.[/]\n"
        )

    text = render_markdown(
        result, shortlist=shortlist_rows, config_path=config.resolve()
    )

    path = write_digest(text, cfg.resolve(cfg.output.dir))
    console.print(Markdown(text))
    console.print(f"\n[dim]written to {path}[/]")

    if result.reportable:
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
                item.fit.cv_variant.value if item.fit else "-",
            )
        console.print(table)

    if not email:
        console.print("[dim]email skipped: --no-email[/]")
    elif not cfg.output.email.enabled:
        # Never fall through in silence. A disabled emailer used to look
        # exactly like a successful send from the caller's side, which under
        # launchd means nobody finds out until they wonder why the 06:30 mail
        # stopped arriving.
        console.print(f"[yellow]email skipped:[/] {_email_skip_reason(cfg)}")
    else:
        try:
            if send_email(text, cfg.output.email, html_body=render_html(text)):
                console.print("[green]emailed[/]")
        except Exception as e:
            # The digest is already on disk, so nothing is lost - but the run
            # did not do what it was scheduled to do, and launchd only records
            # that if the exit status says so.
            console.print(f"[red]email failed:[/] {e}")
            console.print(f"[dim]the digest is still on disk at {path}[/]")
            raise typer.Exit(1) from e


@app.command()
def discover(
    config: ConfigOpt = Path("config.yaml"),
    verbose: VerboseOpt = False,
) -> None:
    """Probe every configured board and report which slugs actually work."""
    _setup_logging(verbose)
    cfg = _load(config)

    async def probe() -> list[tuple[SourceEntry, ProbeResult]]:
        async with Fetcher(cfg.http) as fetcher:

            async def one(entry: SourceEntry) -> tuple[SourceEntry, ProbeResult]:
                return entry, await get_source(entry, fetcher).probe()

            return list(await asyncio.gather(*(one(e) for e in cfg.sources)))

    rows = asyncio.run(probe())

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
    console.print(table)

    verified = sum(1 for _, r in rows if r.trustworthy)
    unknown = sum(1 for _, r in rows if r.status is ProbeStatus.UNKNOWN)
    skipped = sum(1 for _, r in rows if r.status is ProbeStatus.SKIPPED)
    failed = sum(1 for _, r in rows if r.status is ProbeStatus.FAIL)
    console.print(
        f"\n[green]{verified} verified[/], [yellow]{unknown} unverifiable[/], "
        f"[dim]{skipped} skipped[/], [red]{failed} broken[/]  "
        f"(of {len(rows)} configured)"
    )
    if unknown:
        console.print(
            "\n[yellow]UNKNOWN[/] means the API answered 200 with zero rows and "
            "does not 404 unknown slugs, so an empty board and a wrong slug are\n"
            "indistinguishable. Open the careers page in a browser to settle it."
        )
    if failed:
        console.print("\n[bold]Where the slug comes from:[/]")
        kinds = {e.kind for e, r in rows if r.status is ProbeStatus.FAIL}
        for name, cls in sorted(available().items()):
            if name in kinds and cls.slug_hint:
                console.print(f"  [cyan]{name:16}[/] {cls.slug_hint}")


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
def cvs(config: ConfigOpt = Path("config.yaml")) -> None:
    """Show which CV variants were found and how they parse."""
    cfg = _load(config)
    directory = cfg.resolve(cfg.profile.cv_dir) if cfg.profile.cv_dir else None
    library = CVLibrary.load(directory)
    if not library:
        console.print(f"[yellow]No CV variants found in {directory}[/]")
        raise typer.Exit(1)
    table = Table(title=f"CV variants in {directory}", show_edge=False)
    table.add_column("Variant")
    table.add_column("Chars", justify="right")
    table.add_column("Opens with")
    for variant, text in library.variants.items():
        table.add_row(variant.value, str(len(text)), text[:70].replace("\n", " "))
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


@app.command()
def prune(
    config: ConfigOpt = Path("config.yaml"),
    days: Annotated[
        int, typer.Option(help="Drop cached verdicts older than this.")
    ] = 180,
) -> None:
    """Drop stale cached LLM verdicts."""
    cfg = _load(config)

    async def go() -> int:
        async with Store(cfg.resolve(cfg.output.db_path)) as store:
            return await store.prune(days)

    console.print(f"Removed {asyncio.run(go())} cached verdicts.")


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


if __name__ == "__main__":
    app()
