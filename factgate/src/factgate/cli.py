"""The `factgate` command."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import shutil
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from pydantic import ValidationError

from factgate.backends import get_backend
from factgate.cache import Cache
from factgate.config import Config, load_config
from factgate.eval.cases import Case, load_cases
from factgate.eval.compare import compare_markdown
from factgate.eval.live import EvalRun, live, offline_run, regrade
from factgate.eval.report import report_markdown
from factgate.items import read_items
from factgate.output import summary_markdown, write_results
from factgate.pipeline import Result, RunStats, load_resolvers, prefilter_reason, run
from factgate.schema import build_model, system_prompt

app = typer.Typer(
    help="Quoted facts from any text, decided by rules you write.", no_args_is_help=True
)
TEMPLATES = Path(__file__).parent / "templates"
T = TypeVar("T")

ConfigArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="The config YAML.")]
AsOf = Annotated[str | None, typer.Option(help="The date rules treat as today (YYYY-MM-DD).")]


def config_or_exit(path: Path) -> Config:
    try:
        return load_config(path)
    except (ValidationError, ValueError) as e:
        typer.echo(f"{path}: {e}", err=True)
        raise typer.Exit(2) from e


def as_of_date(value: str | None) -> dt.date:
    return dt.date.fromisoformat(value) if value else dt.datetime.now(dt.UTC).date()


def or_exit(where: Path, read: Callable[[], T]) -> T:
    """`read()`, with a mistake in the user's files printed as one line, exit 2."""
    try:
        return read()
    except ValueError as e:
        typer.echo(f"{where}: {e}", err=True)
        raise typer.Exit(2) from e


@app.command()
def check(config: ConfigArg) -> None:
    """Validate a config and print the prompt and schema the model gets."""
    cfg = config_or_exit(config)
    or_exit(config, lambda: load_resolvers(cfg.resolvers, cfg.base))
    typer.echo(system_prompt(cfg.facts))
    typer.echo(json.dumps(build_model(cfg.facts).model_json_schema(), indent=2))


@app.command("run")
def run_cmd(
    config: ConfigArg,
    replay: Annotated[Path | None, typer.Option(help="Answers recorded with --record.")] = None,
    record: Annotated[Path | None, typer.Option(help="Append every answer to this file.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Results JSONL (default out/results.jsonl).")
    ] = None,
    dry: Annotated[bool, typer.Option(help="Stop before any model call.")] = False,
    no_cache: Annotated[bool, typer.Option(help="Neither read nor write the cache.")] = False,
    as_of: AsOf = None,
) -> None:
    """Decide every item the config names."""
    cfg = config_or_exit(config)
    items = or_exit(config, lambda: read_items(cfg.items, cfg.base))
    if dry:
        sending = [i for i in items if prefilter_reason(i, cfg.prefilter) is None]
        typer.echo(f"{len(items)} items; would send {len(sending)} (cache permitting)")
        return
    resolvers = or_exit(config, lambda: load_resolvers(cfg.resolvers, cfg.base))
    backend = get_backend(cfg.model, replay)

    async def go() -> tuple[list[Result], RunStats]:
        async with AsyncExitStack() as stack:
            cache = None
            if cfg.cache_path is not None and not no_cache and replay is None:
                cache = await stack.enter_async_context(Cache(cfg.cache_path))
            return await run(
                cfg,
                items,
                backend,
                as_of=as_of_date(as_of),
                cache=cache,
                record=record,
                resolvers=resolvers,
            )

    results, stats = asyncio.run(go())
    path = out or cfg.base / "out" / "results.jsonl"
    write_results(results, path)
    summary = summary_markdown(cfg, results, stats)
    path.with_suffix(".md").write_text(summary, encoding="utf-8")
    typer.echo(summary)
    typer.echo(f"written to {path}")
    if any(r.status == "error" for r in results):
        raise typer.Exit(1)


@app.command()
def init(folder: Path) -> None:
    """Write a starter config, three items, recorded answers and two cases."""
    if folder.exists() and any(folder.iterdir()):
        typer.echo(f"{folder} is not empty; choose a new folder", err=True)
        raise typer.Exit(2)
    shutil.copytree(TEMPLATES / "starter", folder, dirs_exist_ok=True)
    typer.echo(
        f"wrote {folder}; next: factgate run {folder / 'config.yaml'} --replay "
        f"{folder / 'answers.jsonl'}"
    )


eval_app = typer.Typer(help="Graded cases: offline, live with repeats, compare, regrade.")
app.add_typer(eval_app, name="eval")
CasesArg = Annotated[list[Path], typer.Argument(help="Case files (JSONL or YAML).")]
HeldOut = Annotated[list[Path] | None, typer.Option(help="Held-out case files.")]
ResultsDir = Annotated[Path | None, typer.Option(help="Where runs are written.")]


def _cases(cfg: Config, files: list[Path], heldout: list[Path] | None) -> list[Case]:
    try:
        return load_cases(files, cfg, heldout=heldout or [])
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e


def _save(run: EvalRun, cfg: Config, results: Path | None) -> Path:
    folder = results or cfg.base / "results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run.label}.json"
    path.write_text(run.model_dump_json(indent=1) + "\n", encoding="utf-8")
    report = report_markdown(run, cfg)
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    typer.echo(report)
    typer.echo(f"written to {path}")
    return path


@eval_app.command("run")
def eval_run(
    config: ConfigArg,
    cases: CasesArg,
    heldout: HeldOut = None,
    live_: Annotated[bool, typer.Option("--live", help="Ask the model.")] = False,
    repeats: Annotated[int, typer.Option(min=1, help="Attempts per case (live).")] = 3,
    label: Annotated[str | None, typer.Option(help="Name of the run.")] = None,
    results: ResultsDir = None,
    as_of: AsOf = None,
) -> None:
    """Grade every case. Offline (default) feeds each case's model_facts
    through the code; --live asks the model --repeats times per case."""
    cfg = config_or_exit(config)
    loaded = _cases(cfg, cases, heldout)
    or_exit(config, lambda: load_resolvers(cfg.resolvers, cfg.base))
    today = as_of_date(as_of)
    if live_:
        backend = get_backend(cfg.model, None)
        name = label or f"{today}-{cfg.model.name}"
        run = asyncio.run(live(loaded, cfg, backend, repeats=repeats, label=name, as_of=today))
        _save(run, cfg, results)
        return
    run = offline_run(loaded, cfg, label or "offline", today)
    live_only = len(loaded) - len(run.cases)
    if live_only:
        typer.echo(f"{live_only} case(s) have no model_facts: they run with --live only")
    _save(run, cfg, results)
    problems = run.unexpected()
    if problems:
        typer.echo("CI would fail: " + "; ".join(problems), err=True)
        raise typer.Exit(1)


@eval_app.command("compare")
def eval_compare(
    before: Annotated[Path, typer.Argument(exists=True)],
    after: Annotated[Path, typer.Argument(exists=True)],
) -> None:
    """Which cases moved between two runs; only stable moves count."""
    a = EvalRun.model_validate_json(before.read_text(encoding="utf-8"))
    b = EvalRun.model_validate_json(after.read_text(encoding="utf-8"))
    report = compare_markdown(a, b)
    path = after.parent / f"compare-{a.label}-{b.label}.md"
    path.write_text(report, encoding="utf-8")
    typer.echo(report)


@eval_app.command("regrade")
def eval_regrade(
    run_file: Annotated[Path, typer.Argument(exists=True, help="A saved live run.")],
    config: ConfigArg,
    cases: CasesArg,
    heldout: HeldOut = None,
    results: ResultsDir = None,
    as_of: AsOf = None,
) -> None:
    """Re-score a run's saved answers with today's config and cases, on the
    run's own date unless --as-of says otherwise: no model calls."""
    cfg = config_or_exit(config)
    run = EvalRun.model_validate_json(run_file.read_text(encoding="utf-8"))
    again = regrade(run, _cases(cfg, cases, heldout), cfg, as_of_date(as_of or run.as_of or None))
    _save(again, cfg, results or run_file.parent)
