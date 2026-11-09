"""String pattern helpers: id-likeness, prefixes, email domains, URL hosts, numeric parsing."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

__all__ = ["email_domain", "id_prefix", "is_id_like", "try_number", "url_host"]

_PREFIXED = re.compile(r"^[A-Za-z]{1,6}_[A-Za-z0-9]{8,}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SFID = re.compile(r"^[0-9a-zA-Z]{15}([0-9a-zA-Z]{3})?$")
_TICKET = re.compile(r"^([A-Z]{1,4}-)\d{6,}$")
_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _is_sfid(s: str) -> bool:
    # A bare 15/18-char alnum string is only an id if its 3-char key prefix has a digit
    # (0015..., 003..., 00Q..., a0B...); this keeps ordinary 15-letter words out.
    return _SFID.match(s) is not None and any(c.isdigit() for c in s[:3])


def is_id_like(s: str) -> bool:
    return bool(
        _PREFIXED.match(s) or _UUID.match(s) or _EMAIL.match(s) or _TICKET.match(s) or _is_sfid(s)
    )


def id_prefix(s: str) -> str | None:
    if _UUID.match(s) or _EMAIL.match(s):
        return None
    if _PREFIXED.match(s):
        return s[: s.index("_") + 1]
    if (m := _TICKET.match(s)) is not None:
        return m.group(1)
    if _is_sfid(s):
        return s[:3]
    return None


def _idna(host: str) -> str:
    host = host.lower()
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def email_domain(s: str) -> str | None:
    if not _EMAIL.match(s):
        return None
    return _idna(s.rsplit("@", 1)[1])


def url_host(s: str) -> str | None:
    if not s.lower().startswith(("http://", "https://")):
        return None
    try:
        host = urlsplit(s).hostname
    except ValueError:
        return None
    return _idna(host) if host else None


def try_number(v: object) -> float | None:
    """Parse ``v`` as a number, or ``None``. Never raises.

    Bools are not numbers. Ints too large for a float saturate to ``+/-inf`` (as does
    ``"1e999"``); callers treat non-finite values separately.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        try:
            return float(v)
        except OverflowError:  # int too large for a float: saturate to +/-inf
            return float("inf") if v > 0 else float("-inf")
    if isinstance(v, float):
        return v
    if isinstance(v, str) and _NUMERIC.match(v.strip()):
        return float(v)
    return None
