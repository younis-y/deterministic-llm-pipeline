"""The `factgate` command."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from factgate.backends import get_backend
from factgate.cache import Cache
from factgate.config import Config, load_config
from factgate.items import read_items
from factgate.output import summary_markdown, write_results
from factgate.pipeline import Result, RunStats, load_resolvers, prefilter_reason, run
from factgate.schema import build_model, system_prompt

app = typer.Typer(
    help="Quoted facts from any text, decided by rules you write.", no_args_is_help=True
)
TEMPLATES = Path(__file__).parent / "templates"

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


@app.command()
def check(config: ConfigArg) -> None:
    """Validate a config and print the prompt and schema the model gets."""
    cfg = config_or_exit(config)
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
    items = read_items(cfg.items, cfg.base)
    if dry:
        sending = [i for i in items if prefilter_reason(i, cfg.prefilter) is None]
        typer.echo(f"{len(items)} items; would send {len(sending)} (cache permitting)")
        return
    resolvers = load_resolvers(cfg.resolvers)
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
