from pathlib import Path

import pytest

from factgate.items import Item, ItemsConfig, read_items


def test_jsonl_with_named_fields(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(
        '{"no": "1", "subject": "Hi", "body": "Pay me", "who": "x"}\n\n'
        '{"no": "2", "subject": "Re", "body": "Paid"}\n'
    )
    cfg = ItemsConfig(jsonl="a.jsonl", id="no", fields=["subject", "body"], meta=["who"])
    items = read_items(cfg, tmp_path)
    assert [i.id for i in items] == ["1", "2"]
    assert items[0].fields == {"subject": "Hi", "body": "Pay me"}
    assert items[0].meta == {"who": "x"}
    assert items[1].meta == {}


def test_csv(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("id,text\n7,hello world\n")
    items = read_items(ItemsConfig(csv="a.csv"), tmp_path)
    assert items == [Item(id="7", fields={"text": "hello world"})]


def test_folder(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("bee")
    (tmp_path / "docs" / "a.txt").write_text("ay")
    (tmp_path / "docs" / "skip.pdf").write_text("no")
    items = read_items(ItemsConfig(folder="docs"), tmp_path)
    assert [(i.id, i.fields["text"]) for i in items] == [("a", "ay"), ("b", "bee")]


def test_duplicate_id_is_named(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"id": "x", "text": "1"}\n{"id": "x", "text": "2"}\n')
    with pytest.raises(ValueError, match="duplicate item id 'x'"):
        read_items(ItemsConfig(jsonl="a.jsonl"), tmp_path)


def test_missing_field_is_named(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"id": "x", "body": "1"}\n')
    with pytest.raises(ValueError, match="item 'x' has no field 'text'"):
        read_items(ItemsConfig(jsonl="a.jsonl"), tmp_path)


def test_exactly_one_reader() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        ItemsConfig(jsonl="a", csv="b")


def test_content_hash_depends_on_fields_only() -> None:
    a = Item(id="1", fields={"text": "x"}, meta={"m": 1})
    b = Item(id="2", fields={"text": "x"})
    assert a.content_hash == b.content_hash
    assert a.content_hash != Item(id="1", fields={"text": "y"}).content_hash


def test_a_duplicate_id_from_a_folder_is_named(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "acme.txt").write_text("one")
    (tmp_path / "docs" / "acme.md").write_text("two")
    with pytest.raises(ValueError, match="duplicate item id 'acme'"):
        read_items(ItemsConfig(folder="docs"), tmp_path)


def test_a_csv_saved_with_a_byte_order_mark(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("﻿id,text\n7,hello\n", encoding="utf-8")
    assert read_items(ItemsConfig(csv="a.csv"), tmp_path)[0].id == "7"


def test_content_hash_depends_on_field_order() -> None:
    a = Item(id="1", fields={"subject": "x", "body": "y"})
    b = Item(id="1", fields={"body": "y", "subject": "x"})
    assert a.content_hash != b.content_hash
