"""What a run writes: one JSON line per item, and a short Markdown summary."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from factgate.config import Config
from factgate.pipeline import Result, RunStats


def write_results(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() for r in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(title: str, counts: Counter[str]) -> list[str]:
    rows = [f"| {key} | {n} |" for key, n in counts.most_common()]
    return [f"| {title} | Items |", "|---|---|", *rows, ""] if rows else []


def _calls(stats: RunStats) -> str:
    if stats.backend == "replay":
        return (
            f"{stats.calls} answers replayed from a file, no model called; "
            f"{stats.cache_hits} from cache."
        )
    return (
        f"{stats.calls} model calls, {stats.cache_hits} from cache, "
        f"{stats.input_tokens:,} input and {stats.output_tokens:,} output tokens, "
        f"{stats.seconds:g} seconds."
    )


def summary_markdown(cfg: Config, results: list[Result], stats: RunStats) -> str:
    decided = [r for r in results if r.status == "decided"]
    lines = [
        f"# {cfg.name}",
        "",
        f"{stats.items} items: {len(decided)} decided, {stats.filtered} filtered, "
        f"{stats.deferred} deferred, {stats.errors} errors.",
        _calls(stats),
        "",
        *_table("Outcome", Counter(str(r.outcome) for r in decided)),
        *_table("Rule", Counter(r.rule or f"(default: {cfg.default_outcome})" for r in decided)),
    ]
    problems = [r for r in results if r.status in {"error", "deferred"}]
    if problems:
        lines += ["| Item | Status | Why |", "|---|---|---|"]
        lines += [f"| {r.id} | {r.status} | {r.reason} |" for r in problems]
    return "\n".join(lines).rstrip() + "\n"
