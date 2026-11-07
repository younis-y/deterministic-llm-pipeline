from pathlib import Path

from factgate.cache import Cache, cache_key, fingerprint
from factgate.items import Item


async def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "c.sqlite"
    async with Cache(path) as cache:
        assert await cache.get("k") is None
        await cache.put("k", {"a": 1})
    async with Cache(path) as cache:
        assert await cache.get("k") == {"a": 1}


def test_key_names_item_prompt_backend_and_model() -> None:
    item = Item(id="1", fields={"text": "x"})
    fp = fingerprint("sys", "{}", 100, None)
    assert fp != fingerprint("sys2", "{}", 100, None)
    assert fp != fingerprint("sys", "{}", 100, 0.0)
    assert fp != fingerprint("sys", "{}", 100, None, "http://other:11434/v1")
    assert len(fp) == 12
    assert cache_key(item, fp, "anthropic", "m") == f"{item.content_hash}:{fp}:anthropic:m"
