"""Graded cases for the facts pipeline: the code alone, or a live model.

Each case in `cases.json` is a synthetic advert and what the finished facts,
and the rule `decide` fires, must be. Offline (`tests/test_eval_cases.py`, in
CI) a case's `model_facts`, a realistic model answer, go through the same
verification, resolvers and rules a scan uses; a case with none tests the
code readers on the advert alone. Live, the configured model answers instead,
through `FitScorer` exactly as a scan calls it:

    python -m evals.harness --backend anthropic --model claude-haiku-5-5 --runs 3

prints pass@1 and pass^k per case and writes the run to `evals/results/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rolescan.config import LLMConfig, ProfileConfig, RulesConfig
from rolescan.models import Job, ScoredJob
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import PostingFacts
from rolescan.scoring.llm import finish_facts
from rolescan.scoring.rules import decide

HERE = Path(__file__).parent
SPEC = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = SPEC["cases"]
RULES = RulesConfig(**SPEC["rules"])
MIN_REPORT_SCORE = 40
#: The candidate the live prompt describes; the cases' rules are written for it.
SUMMARY = (
    "Recent MSc graduate looking for graduate or junior roles in data "
    "engineering, data science, machine learning and quantitative analysis. "
    "Python and SQL."
)


def job(case: dict[str, Any]) -> Job:
    return Job(
        source="eval",
        company="Example",
        title=case["title"],
        location="London",
        url=f"https://example.invalid/{case['id']}",
        description=case["description"],
    )


def _value(facts: PostingFacts, name: str) -> Any:
    if name == "hard_bars":
        return sorted({bar.kind.value for bar in facts.hard_bars})
    value = getattr(facts, name).value
    return getattr(value, "value", value)


def grade(case: dict[str, Any], facts: PostingFacts, rule: str | None) -> list[str]:
    """Why `case` failed with these finished facts and rule, or [] if it passed."""
    expect = case["expect"]
    got = _value(facts, expect["fact"]) if "fact" in expect else None
    why = []
    if "value" in expect and got != expect["value"]:
        why.append(f"{expect['fact']}={got!r}")
    if "value_max" in expect and got is not None and got > expect["value_max"]:
        why.append(f"{expect['fact']}={got!r}")
    if "value_in" in expect and not set(got or []) & set(expect["value_in"]):
        why.append(f"{expect['fact']}={got!r}")
    if "rule" in expect and rule != expect["rule"]:
        why.append(f"rule={rule}")
    if "rule_not" in expect and rule == expect["rule_not"]:
        why.append(f"rule {rule} fired")
    return why


def offline(case: dict[str, Any]) -> list[str]:
    """`grade` with the case's own `model_facts` standing in for the model."""
    raw = PostingFacts.model_validate(
        {"fit_score": 80, "reason": "fixture", **case.get("model_facts", {})}
    )
    facts = finish_facts(raw, job(case))
    return grade(case, facts, decide(facts, RULES, MIN_REPORT_SCORE).rule)


async def live_run(cfg: LLMConfig) -> tuple[dict[str, list[str]], dict[str, int]]:
    """One pass of every case through the model: each case's failures, and
    the token counts the backend reported (Anthropic only)."""
    profile = ProfileConfig(
        summary=SUMMARY, rules=RULES, min_report_score=MIN_REPORT_SCORE
    )
    scorer = FitScorer(cfg, profile)  # no store: every case asks the model
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    judge = scorer._get_judge()
    parse = getattr(judge, "_parse", None)
    if parse is not None:

        async def counted(*args: Any, **kwargs: Any) -> Any:
            response = await parse(*args, **kwargs)
            u = response.usage
            usage["input"] += u.input_tokens
            usage["output"] += u.output_tokens
            usage["cache_read"] += u.cache_read_input_tokens or 0
            usage["cache_write"] += u.cache_creation_input_tokens or 0
            return response

        setattr(judge, "_parse", counted)  # noqa: B010 - an eval-only meter
    scored = await scorer.score_all(
        [ScoredJob(job=job(c), keyword_score=50) for c in CASES]
    )
    by_url = {s.job.url: s for s in scored}
    out = {}
    for case in CASES:
        url = job(case).url
        fit, facts = by_url[url].fit, scorer.last_facts.get(url)
        out[case["id"]] = (
            grade(case, facts, fit.rule)
            if fit is not None and facts is not None
            else [f"no answer ({scorer.first_error or 'deferred'})"]
        )
    return out, usage


async def main(backend: str, model: str, runs: int) -> None:
    cfg = LLMConfig(
        enabled=True, backend=backend, model=model, max_calls_per_run=len(CASES)
    )
    started = time.monotonic()
    passes = [await live_run(cfg) for _ in range(runs)]
    seconds = round(time.monotonic() - started)
    rows = []
    for case in CASES:
        fails = [p[0][case["id"]] for p in passes]
        rows.append(
            {
                "id": case["id"],
                "passed": sum(not f for f in fails),
                "runs": runs,
                "offline_passes": not offline(case),
                "open_defect": case.get("open_defect", ""),
                "failures": [f for f in fails if f],
            }
        )
        mark = "ok  " if all(not f for f in fails) else "FAIL"
        print(f"{mark} {case['id']:14} {rows[-1]['passed']}/{runs}  {fails[0] or ''}")
    usage = {k: sum(p[1][k] for p in passes) for k in passes[0][1]}
    pass_1 = sum(r["passed"] for r in rows) / (len(rows) * runs)
    pass_k = sum(r["passed"] == runs for r in rows) / len(rows)
    print(f"\npass@1 {pass_1:.0%}  pass^{runs} {pass_k:.0%}  {seconds}s  {usage}")
    result = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": backend,
        "model": model,
        "runs": runs,
        "pass_at_1": round(pass_1, 3),
        f"pass_hat_{runs}": round(pass_k, 3),
        "seconds": seconds,
        "tokens": usage,
        "cases": rows,
    }
    path = HERE / "results" / f"{result['date'][:10]}-{model}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="anthropic")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.backend, args.model, args.runs))
