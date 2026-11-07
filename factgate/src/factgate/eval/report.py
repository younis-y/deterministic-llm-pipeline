"""The Markdown report of one eval run, and a check that a README quotes it."""

from __future__ import annotations

from collections import Counter

from factgate.config import Config
from factgate.eval.live import CaseRun, EvalRun, wilson

STATES = ("pass", "fail", "flaky", "inconclusive")


def _counts(cases: list[CaseRun]) -> str:
    c = Counter(x.state for x in cases)
    return " | ".join(str(c[s]) for s in STATES)


def _why(case: CaseRun) -> str:
    misses = {
        "; ".join(a.fact_failures + ([a.outcome_failure] if a.outcome_failure else [])) or a.error
        for a in case.attempts
        if a.outcome != "pass"
    }
    noise = " (model's facts changed between attempts)" if case.facts_changed else ""
    return " / ".join(sorted(misses)) + noise


def headline(run: EvalRun) -> list[str]:
    """The figures a README should quote for this run."""
    return [f"`{run.model}`, {run.repeats} runs", f"pass@1 {run.pass_at_1:.0%}"]


def report_markdown(run: EvalRun, cfg: Config) -> str:
    visible = [c for c in run.cases if not c.heldout]
    held = [c for c in run.cases if c.heldout]
    conclusive = [c for c in run.cases if c.state != "inconclusive"]
    passed = sum(c.state == "pass" for c in conclusive)
    lo, hi = wilson(passed, len(conclusive))
    lines = [f"# {cfg.name}: {run.label} ({run.mode})", ""]
    dangerous = [c for c in run.cases if any(a.dangerous for a in c.attempts)]
    if dangerous:
        lines += [
            f"**Dangerous mistakes: {len(dangerous)}**: {', '.join(c.id for c in dangerous)}",
            "",
        ]
    lines += [
        f"Model `{run.model or '-'}` via {run.backend or '-'}, {run.repeats} attempt(s) per case, "
        f"{run.seconds:g} s, tokens {run.tokens or '-'}.",
        f"pass@1 {run.pass_at_1:.0%}; pass^{run.repeats} {run.pass_hat_k:.0%} "
        f"({passed} of {len(conclusive)} conclusive cases pass every attempt; "
        f"Wilson 95% {lo:.0%}-{hi:.0%}). Inconclusive cases are outside these figures.",
        "",
        "| Split | Cases | Pass | Fail | Flaky | Inconclusive |",
        "|---|---|---|---|---|---|",
        f"| visible | {len(visible)} | {_counts(visible)} |",
        f"| held-out | {len(held)} | {_counts(held)} |",
        "",
    ]
    tags = sorted({t for c in visible for t in c.tags})
    if tags:
        lines += [
            "| Tag | Cases | Pass | Fail | Flaky | Inconclusive |",
            "|---|---|---|---|---|---|",
        ]
        for tag in tags:
            tagged = [c for c in visible if tag in c.tags]
            lines.append(f"| {tag} | {len(tagged)} | {_counts(tagged)} |")
        lines.append("")
    misses = [c for c in visible if c.state != "pass"]
    if misses:
        lines += ["| Case | State | Why |", "|---|---|---|"]
        for c in misses:
            mark = f" (open defect: {c.open_defect})" if c.open_defect else ""
            lines.append(f"| {c.id}{mark} | {c.state} | {_why(c)} |")
        lines.append("")
    if held:
        lines += ["## Held-out", "", "Ids and states only, so nobody tunes to them.", ""]
        lines += [f"- {c.id}: {c.state}" for c in held]
    return "\n".join(lines).rstrip() + "\n"


def readme_matches(readme: str, run: EvalRun) -> list[str]:
    """The figures from `headline` that `readme` does not quote."""
    return [figure for figure in headline(run) if figure not in readme]
