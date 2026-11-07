"""Eval cases: an item, and what its finished facts and outcome must be.

Case files are JSONL or YAML, one area per file, merged on load. Loading
fails loudly, naming the file and line, on anything that would otherwise
grade silently wrong (from the Judge the Agent hackathon's `merge_cases`).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from factgate.config import Config, check_condition
from factgate.items import Item
from factgate.rules import Condition


class Case(BaseModel):
    #: A misspelt key would grade silently wrong, so it fails to load.
    model_config = ConfigDict(extra="forbid")

    id: str
    fields: dict[str, str]
    expect_facts: dict[str, Condition] = {}
    expect_outcome: str | None = None
    #: A model answer (recorded or hand-written) for the offline run.
    model_facts: dict[str, Any] | None = None
    tags: list[str] = []
    #: A known failure: runs as a strict expected failure until fixed.
    open_defect: str = ""
    #: The expectation is what the code does with THIS `model_facts` (a
    #: misread number the quote check clears, say), not the truth about the
    #: document, so a live model's different answer would grade nothing.
    offline_only: bool = False
    as_of: dt.date | None = None
    note: str = ""
    #: Shown as id and totals only, so nobody tunes to it.
    heldout: bool = False
    #: Where the case was read: `file:line`.
    source: str = ""

    @property
    def item(self) -> Item:
        return Item(id=self.id, fields=self.fields)


def _yaml_lines(node: Any) -> list[int]:
    """The line each case starts on, from the composed YAML."""
    if isinstance(node, yaml.MappingNode):
        node = next((v for k, v in node.value if k.value == "cases"), None)
    if not isinstance(node, yaml.SequenceNode):
        return []
    return [entry.start_mark.line + 1 for entry in node.value]


def _rows(path: Path) -> list[tuple[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text) or []
            lines = _yaml_lines(yaml.compose(text))
        except yaml.YAMLError as e:
            raise ValueError(f"{path}: not valid YAML ({e})") from e
        rows = data.get("cases", []) if isinstance(data, dict) else data
        lines = lines if len(lines) == len(rows) else list(range(1, len(rows) + 1))
        return [(f"{path}:{n}", row) for n, row in zip(lines, rows, strict=True)]
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            try:
                out.append((f"{path}:{n}", json.loads(line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from e
    return out


def _case(row: Any, where: str, cfg: Config, heldout: bool) -> Case:
    if not isinstance(row, dict) or "id" not in row:
        raise ValueError(f"{where}: a case needs an id")
    row = dict(row)
    if "text" in row:
        row["fields"] = {cfg.items.fields[0]: row.pop("text")}
    if not row.get("fields"):
        raise ValueError(f"{where}: no text (give `text:` or `fields:`)")
    expect = row.pop("expect", {}) or {}
    unknown = sorted(set(expect) - {"facts", "outcome"}) if isinstance(expect, dict) else []
    if unknown or not isinstance(expect, dict):
        raise ValueError(f"{where}: expect has {', '.join(map(repr, unknown)) or 'no mapping'}")
    try:
        case = Case(
            **row,
            expect_facts=expect.get("facts", {}),
            expect_outcome=expect.get("outcome"),
            heldout=heldout,
            source=where,
        )
    except ValidationError as e:
        err = e.errors()[0]
        at = ".".join(str(part) for part in err["loc"])
        raise ValueError(f"{where}: {at + ': ' if at else ''}{err['msg']}") from e
    for fact, cond in case.expect_facts.items():
        check_condition(cfg, fact, cond, where)
    if case.offline_only and case.model_facts is None:
        raise ValueError(f"{where}: an offline_only case needs model_facts")
    for fact in case.model_facts or {}:
        if fact not in cfg.facts:
            raise ValueError(f"{where}: model_facts has unknown fact {fact!r}")
    if case.expect_outcome is not None and case.expect_outcome not in cfg.outcomes:
        raise ValueError(f"{where}: unknown outcome {case.expect_outcome!r}")
    if not case.expect_facts and case.expect_outcome is None:
        raise ValueError(f"{where}: nothing to grade (give expect.facts or expect.outcome)")
    return case


def load_cases(paths: list[Path], cfg: Config, *, heldout: list[Path] | None = None) -> list[Case]:
    """Every case in `paths` (visible) and `heldout`, checked against `cfg`."""
    cases: list[Case] = []
    seen: dict[str, str] = {}
    files = [(p, False) for p in paths] + [(p, True) for p in heldout or []]
    for path, hidden in files:
        for where, row in _rows(path):
            case = _case(row, where, cfg, hidden)
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r}: {seen[case.id]} and {where}")
            seen[case.id] = where
            cases.append(case)
    return cases
