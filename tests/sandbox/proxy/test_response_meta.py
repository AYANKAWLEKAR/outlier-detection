"""Bounded response metadata (spec 2.3, D-003) extracted from an MCP tool result."""

from __future__ import annotations

from typing import Any

import orjson
from mcp.types import CallToolResult, ImageContent, TextContent

from agentfw.sandbox.proxy_py import extract_response_meta


def _res(*texts: str, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=t) for t in texts], is_error=is_error
    )


def _json(obj: Any) -> CallToolResult:
    return _res(orjson.dumps(obj).decode())


def test_response_meta_extraction_cases() -> None:
    assert extract_response_meta(_json([1, 2, 3])).item_count == 3
    assert extract_response_meta(_json({"data": [{"a": 1}] * 4})).item_count == 4
    assert extract_response_meta(_json({"totalSize": 7})).item_count == 7
    for key in ("records", "items", "messages", "results", "tickets"):
        assert extract_response_meta(_json({key: [0, 0]})).item_count == 2, key
    # the first listed key wins; a list beats totalSize
    assert extract_response_meta(_json({"records": [0], "totalSize": 9})).item_count == 1
    assert extract_response_meta(_json({"data": [0, 0], "items": [0]})).item_count == 2
    # RM1: only list-shaped results carry a count; a single object (a write's echo) does not
    assert extract_response_meta(_json({"id": "ch_0000000001", "amount": 5})).item_count is None
    assert extract_response_meta(_json({"data": {"x": 1}})).item_count is None
    assert extract_response_meta(_json({"totalSize": True})).item_count is None
    assert extract_response_meta(_json({"updated": 40, "failed": []})).item_count is None
    # JSON scalars and non-JSON text carry no count
    assert extract_response_meta(_json(42)).item_count is None
    assert extract_response_meta(_res("not json {")).item_count is None
    assert extract_response_meta(_json([])).item_count == 0


def test_bytes_cover_all_text_utf8() -> None:
    meta = extract_response_meta(_res("héllo", "[1]"))
    assert meta.bytes == len("héllo".encode()) + 3
    assert meta.item_count is None  # only the first text block is parsed
    empty = extract_response_meta(CallToolResult(content=[]))
    assert (empty.bytes, empty.item_count, empty.ids) == (0, None, ())
    image = CallToolResult(content=[ImageContent(type="image", data="AAAA", mime_type="image/png")])
    assert extract_response_meta(image).bytes == 0


def test_error_flag() -> None:
    meta = extract_response_meta(_res('{"data": [1, 2]}', is_error=True))
    assert meta.error is True and meta.item_count is None
    assert extract_response_meta(_json([1])).error is False


def test_ids_walk_nested_json_dedup_and_cap() -> None:
    payload = {
        "data": [
            {
                "id": "ch_0000000001",
                "customer": {"id": "cus_000000001", "email": "ann@acme.example"},
                "note": "short",
                "amount": 12,
                "ch_keyisnotvalue1": "x",
            },
            {"id": "ch_0000000001", "refs": [["re_0000000001"], "tk_UJH824GX6V"]},
        ]
    }
    meta = extract_response_meta(_json(payload))
    assert meta.ids == (
        "ch_0000000001",
        "cus_000000001",
        "ann@acme.example",
        "re_0000000001",
        "tk_UJH824GX6V",
    )

    many = {"records": [{"Id": f"ch_{i:010d}", "n": [f"cu_{i:010d}"]} for i in range(300)]}
    capped = extract_response_meta(_json(many))
    assert len(capped.ids) == 256
    assert capped.ids[:3] == ("ch_0000000000", "cu_0000000000", "ch_0000000001")


def test_ids_from_non_json_text_are_not_extracted() -> None:
    assert extract_response_meta(_res("charge ch_0000000001 refunded")).ids == ()
