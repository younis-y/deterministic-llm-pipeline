"""Before and after: only stable moves count (the hackathon's `compare`).

A case that went from failing every time to passing every time is fixed, the
reverse is broken. Any move into or out of `flaky` is noise, and a case
inconclusive on either side says nothing. Held-out cases are counted apart.
"""

from __future__ import annotations

from pydantic import BaseModel

from factgate.eval.live import EvalRun


class Moves(BaseModel):
    split: str
    fixed: list[str] = []
    broke: list[str] = []
    noise: list[str] = []
    inconclusive: list[str] = []
    stable_pass_before: int = 0
    stable_pass_after: int = 0
    total: int = 0


def _moves(split: str, before: dict[str, str], after: dict[str, str]) -> Moves:
    moves = Moves(split=split)
    for case in sorted(before.keys() & after.keys()):
        b, a = before[case], after[case]
        moves.total += 1
        moves.stable_pass_before += b == "pass"
        moves.stable_pass_after += a == "pass"
        if "inconclusive" in (b, a):
            moves.inconclusive.append(case)
        elif (b, a) == ("fail", "pass"):
            moves.fixed.append(case)
        elif (b, a) == ("pass", "fail"):
            moves.broke.append(case)
        elif b != a:
            moves.noise.append(case)
    return moves


def compare(before: EvalRun, after: EvalRun) -> tuple[Moves, Moves]:
    """(visible, held-out) moves from `before` to `after`."""
    out = []
    for split, hidden in (("visible", False), ("held-out", True)):
        b = {c.id: c.state for c in before.cases if c.heldout is hidden}
        a = {c.id: c.state for c in after.cases if c.heldout is hidden}
        out.append(_moves(split, b, a))
    return out[0], out[1]


def compare_markdown(before: EvalRun, after: EvalRun) -> str:
    lines = [
        f"# {before.label} -> {after.label}",
        "",
        "| Split | Fixed (fail->pass) | Broke (pass->fail) | Noise (flaky either side) "
        "| Inconclusive either side | Stable pass before -> after |",
        "|---|---|---|---|---|---|",
    ]
    for m in compare(before, after):
        lines.append(
            f"| {m.split} | {len(m.fixed)} | {len(m.broke)} | {len(m.noise)} "
            f"| {len(m.inconclusive)} | {m.stable_pass_before}/{m.total} -> "
            f"{m.stable_pass_after}/{m.total} |"
        )
    visible, _ = compare(before, after)
    for name, ids in (("Fixed", visible.fixed), ("Broke", visible.broke)):
        if ids:
            lines += ["", f"{name} (visible): {', '.join(ids)}"]
    lines += ["", "Quote the transitions, not the totals: a total also moves with noise."]
    return "\n".join(lines) + "\n"
