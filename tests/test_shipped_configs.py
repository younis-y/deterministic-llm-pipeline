from pathlib import Path

import pytest

from rolescan.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [ROOT / "config.example.yaml", *sorted((ROOT / "examples").glob("*.yaml"))],
    ids=lambda path: path.name,
)
def test_every_shipped_yaml_loads(path: Path) -> None:
    Config.load(path)
