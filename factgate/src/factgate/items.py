"""Items: the documents factgate reads, and the readers that load them."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

_READERS = ("jsonl", "csv", "folder", "stdin", "plugin")
_FOLDER_SUFFIXES = {".txt", ".md"}


class Item(BaseModel):
    """One document: an id, one or more named text fields, and pass-through meta.

    Each field is checked separately for quotes, never joined, so a quote
    cannot splice the end of one field onto the start of the next.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    fields: dict[str, str]
    meta: dict[str, Any] = {}

    @property
    def content_hash(self) -> str:
        """The text in field order: reordering fields reorders the prompt."""
        blob = json.dumps(list(self.fields.items()), ensure_ascii=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ItemsConfig(BaseModel):
    """Where items come from: exactly one reader, and which keys hold what."""

    model_config = ConfigDict(extra="forbid")

    jsonl: str | None = None
    csv: str | None = None
    folder: str | None = None
    stdin: bool = False
    #: A `factgate.sources` entry point; it receives this whole config.
    plugin: str | None = None
    options: dict[str, Any] = {}
    id: str = "id"
    fields: list[str] = ["text"]
    meta: list[str] = []

    @model_validator(mode="after")
    def _one_reader(self) -> ItemsConfig:
        chosen = [r for r in _READERS if getattr(self, r)]
        if len(chosen) != 1:
            msg = f"items: give exactly one of {', '.join(_READERS)} (got {chosen or 'none'})"
            raise ValueError(msg)
        if not self.fields:
            raise ValueError("items: fields must name at least one field")
        return self


def _from_rows(rows: Iterable[dict[str, Any]], cfg: ItemsConfig) -> list[Item]:
    items: list[Item] = []
    for row in rows:
        if cfg.id not in row:
            msg = f"an item has no id field {cfg.id!r}: {str(row)[:80]}"
            raise ValueError(msg)
        item_id = str(row[cfg.id])
        missing = [f for f in cfg.fields if f not in row]
        if missing:
            msg = f"item {item_id!r} has no field {missing[0]!r}"
            raise ValueError(msg)
        fields = {f: str(row[f] or "") for f in cfg.fields}
        meta = {m: row[m] for m in cfg.meta if m in row}
        items.append(Item(id=item_id, fields=fields, meta=meta))
    return items


def _jsonl(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for n, line in enumerate(lines, 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"line {n}: not JSON ({e.msg})"
                raise ValueError(msg) from e


def _folder(path: Path, cfg: ItemsConfig) -> list[Item]:
    field = cfg.fields[0]
    return [
        Item(id=p.stem, fields={field: p.read_text(encoding="utf-8")})
        for p in sorted(path.iterdir())
        if p.suffix in _FOLDER_SUFFIXES
    ]


def _plugin(name: str) -> Callable[[ItemsConfig, Path], list[Item]]:
    found = entry_points(group="factgate.sources", name=name)
    if not found:
        msg = f"no factgate.sources plugin named {name!r}"
        raise ValueError(msg)
    reader: Callable[[ItemsConfig, Path], list[Item]] = next(iter(found)).load()
    return reader


def read_items(cfg: ItemsConfig, base: Path) -> list[Item]:
    """Every item the config names; relative paths are read from `base`. Two
    items with one id would share one result, so that is an error."""
    items = _read(cfg, base)
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate item id {item.id!r}")
        seen.add(item.id)
    return items


def _read(cfg: ItemsConfig, base: Path) -> list[Item]:
    if cfg.jsonl:
        text = (base / cfg.jsonl).read_text(encoding="utf-8")
        return _from_rows(_jsonl(text.splitlines()), cfg)
    if cfg.csv:
        with (base / cfg.csv).open(newline="", encoding="utf-8-sig") as f:
            return _from_rows(csv.DictReader(f), cfg)
    if cfg.folder:
        return _folder(base / cfg.folder, cfg)
    if cfg.stdin:
        return _from_rows(_jsonl(sys.stdin), cfg)
    assert cfg.plugin is not None  # the validator allows nothing else
    return _plugin(cfg.plugin)(cfg, base)
