from __future__ import annotations

import pytest

from agentfw.core.patterns import email_domain, id_prefix, is_id_like, try_number, url_host


@pytest.mark.parametrize(
    "s",
    [
        "ch_3Nabc12345XYZ",
        "acct_1234567890",
        "123e4567-e89b-12d3-a456-426614174000",
        "bob@acme.example",
        "003Dn00000ABCDEFGH",
        "003Dn00000ABCDE",
        "T-00000001",
    ],
)
def test_is_id_like_positive(s: str) -> None:
    assert is_id_like(s)


@pytest.mark.parametrize("s", ["hello", "ch_short", "USD", "internationally", "", "37.50"])
def test_is_id_like_negative(s: str) -> None:
    assert not is_id_like(s)


def test_id_prefix() -> None:
    assert id_prefix("ch_3Nabc12345XYZ") == "ch_"
    assert id_prefix("acct_1234567890") == "acct_"
    assert id_prefix("003Dn00000ABCDEFGH") == "003"
    assert id_prefix("123e4567-e89b-12d3-a456-426614174000") is None
    assert id_prefix("bob@acme.example") is None
    assert id_prefix("plain") is None


def test_email_domain_and_url_host() -> None:
    assert email_domain("Bob@ACME.example") == "acme.example"
    assert email_domain("not an email") is None
    assert email_domain("a@BÜCHER.example") == "xn--bcher-kva.example"
    assert url_host("https://Files.Example.com:8080/x?y=1") == "files.example.com"
    assert url_host("ftp://x.example") is None
    assert url_host("a@..example") is None


def test_try_number() -> None:
    assert try_number(3) == 3.0
    assert try_number(2.5) == 2.5
    assert try_number("37.50") == 37.5
    assert try_number("-1e3") == -1000.0
    assert try_number(True) is None
    assert try_number("abc") is None
    assert try_number("nan") is None
    assert try_number(None) is None


def test_try_number_huge_int_does_not_overflow() -> None:
    # Review Focus 2: float(10**400) raises OverflowError; ints too large for float map to +/-inf.
    assert try_number(10**400) == float("inf")
    assert try_number(-(10**400)) == float("-inf")
    assert try_number("1" + "0" * 400) == float("inf")
    assert try_number("1e999") == float("inf")
