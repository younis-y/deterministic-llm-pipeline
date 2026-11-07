"""The one YAML file that says what to read, what to extract and how to decide."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from factgate.items import ItemsConfig
from factgate.rules import ALLOWED, OPERATORS, Condition, Rule
from factgate.schema import FactSpec


class Prefilter(BaseModel):
    """Plain keyword rules applied before any model call."""

    model_config = ConfigDict(extra="forbid")

    require_any: list[str] = []
    exclude_any: list[str] = []


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "anthropic"
    name: str
    base_url: str | None = None
    #: Env var holding the API key; the backend's own default when unset.
    api_key_env: str | None = None
    max_calls: int = 100
    max_minutes: float = 30
    max_concurrent: int = 4
    max_tokens: int = 2048
    #: Sent only when set: some models refuse the parameter.
    temperature: float | None = None


class Dangerous(BaseModel):
    """A mistake that costs more than others: `expected` decided as `got`."""

    model_config = ConfigDict(extra="forbid")

    expected: str
    got: str


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    items: ItemsConfig
    prefilter: Prefilter | None = None
    facts: dict[str, FactSpec]
    rules: list[Rule] = []
    default_outcome: str
    model: ModelConfig
    cache: str | None = ".factgate/cache.sqlite"
    #: `module:function` names, run in order on the checked facts.
    resolvers: list[str] = []
    dangerous: list[Dangerous] = []
    #: The folder the config was read from; relative paths start here.
    base: Path = Path()

    @property
    def outcomes(self) -> set[str]:
        return {self.default_outcome, *(r.then for r in self.rules)}

    @property
    def cache_path(self) -> Path | None:
        return self.base / self.cache if self.cache else None

    @model_validator(mode="after")
    def _check(self) -> Config:
        for rule in self.rules:
            if not rule.when:
                raise ValueError(f"rule {rule.name!r}: `when` is empty")
            for fact, cond in rule.when.items():
                check_condition(self, fact, cond, f"rule {rule.name!r}")
        for d in self.dangerous:
            for outcome in (d.expected, d.got):
                if outcome not in self.outcomes:
                    raise ValueError(f"dangerous: {outcome!r} is not an outcome")
        return self


def check_condition(cfg: Config, fact: str, cond: Condition, where: str) -> None:
    """Raise, naming `where`, if `cond` cannot apply to `fact` (rules and cases)."""
    if fact not in cfg.facts:
        raise ValueError(f"{where}: unknown fact {fact!r}")
    if not isinstance(cond, dict) or not cond:
        raise ValueError(f"{where}: the condition on {fact!r} must be a mapping of operators")
    spec = cfg.facts[fact]
    for op, arg in cond.items():
        if op not in OPERATORS:
            raise ValueError(f"{where}: unknown operator {op!r}")
        if spec.type == "list" and op in {"contains", "stated"}:
            continue
        if op not in ALLOWED[spec.scalar]:
            raise ValueError(f"{where}: {op!r} does not apply to a {spec.scalar} fact")
        if op == "matches":
            try:
                re.compile(arg)
            except re.error as e:
                raise ValueError(f"{where}: bad regex {arg!r} ({e})") from e


def load_config(path: Path) -> Config:
    """Read and check a config; any mistake fails here, before a model call."""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a YAML mapping")
    return Config.model_validate({**data, "base": path.resolve().parent})
