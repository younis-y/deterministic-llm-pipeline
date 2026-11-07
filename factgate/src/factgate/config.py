"""The one YAML file that says what to read, what to extract and how to decide."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    max_calls: Annotated[int, Field(ge=0)] = 100
    max_minutes: Annotated[float, Field(gt=0)] = 30
    max_concurrent: Annotated[int, Field(ge=1)] = 4
    max_tokens: Annotated[int, Field(ge=1)] = 2048
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
        listed = spec.type == "list" and op in {"contains", "stated"}
        if not listed and op not in ALLOWED[spec.scalar]:
            raise ValueError(f"{where}: {op!r} does not apply to a {spec.scalar} fact")
        problem = _argument_problem(spec, op, arg)
        if problem:
            raise ValueError(f"{where}: {op!r} on {fact!r} {problem} (got {arg!r})")


def _argument_problem(spec: FactSpec, op: str, arg: Any) -> str | None:
    """Why `arg` cannot be what `op` compares this fact with, or None."""
    if op == "stated":
        return None if isinstance(arg, bool) else "needs true or false"
    if op in {"within_days", "before_days"}:
        whole = isinstance(arg, int) and not isinstance(arg, bool) and arg >= 0
        return None if whole else "needs a whole number of days, 0 or more"
    if op == "in":
        if not isinstance(arg, list) or not arg:
            return "needs a list"
        return next((p for a in arg if (p := _value_problem(spec, a))), None)
    if op == "matches":
        if not isinstance(arg, str):
            return "needs text"
        try:
            re.compile(arg)
        except re.error as e:
            return f"is a bad regex ({e})"
        return None
    if op == "contains" and spec.type != "list":
        return None if isinstance(arg, str) else "needs text"
    return _value_problem(spec, arg)


def _value_problem(spec: FactSpec, arg: Any) -> str | None:
    """Why `arg` is not a value this fact can take, or None."""
    kind = spec.scalar
    if kind in {"number", "integer"}:
        number = isinstance(arg, int | float) and not isinstance(arg, bool)
        return None if number else "needs a number"
    if kind == "date":
        if isinstance(arg, dt.date):
            return None
        try:
            dt.date.fromisoformat(str(arg))
        except ValueError:
            return "needs a date (YYYY-MM-DD)"
        return None
    if kind == "bool":
        return None if isinstance(arg, bool) else "needs true or false"
    if kind == "enum":
        values = spec.values or []
        return None if arg in values else f"needs one of {', '.join(values)}"
    return None if isinstance(arg, str) else "needs text"


def load_config(path: Path) -> Config:
    """Read and check a config; any mistake fails here, before a model call."""
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: not valid YAML ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a YAML mapping")
    return Config.model_validate({**data, "base": path.resolve().parent})
