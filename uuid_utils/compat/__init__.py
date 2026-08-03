"""
uuid_utils.compat — the surface langchain_core and langsmith actually import.

Both do exactly this and nothing else:

    from uuid_utils.compat import uuid7

The real package's compat layer returns stdlib `uuid.UUID` objects rather than
its own native UUID type; these do the same, since they are stdlib UUIDs
already. See ../__init__.py for why this shim exists.
"""

from __future__ import annotations

from uuid import (
    NAMESPACE_DNS,
    NAMESPACE_OID,
    NAMESPACE_URL,
    NAMESPACE_X500,
    RESERVED_FUTURE,
    RESERVED_MICROSOFT,
    RESERVED_NCS,
    RFC_4122,
    UUID,
    SafeUUID,
    getnode,
)

import uuid_utils as _u
from uuid_utils import _uuid4_int, _uuid7_int

NIL = UUID("00000000-0000-0000-0000-000000000000")
MAX = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")


def uuid7(timestamp: int | None = None, nanos: int = 0) -> UUID:
    """Time-ordered UUID, monotonic within a millisecond (RFC 9562 §5.7).

    langchain_core calls this both bare and as uuid7(timestamp=s, nanos=ns),
    so both forms must work.
    """
    return UUID(int=_uuid7_int(timestamp, nanos))


def uuid1(node=None, clock_seq=None) -> UUID:
    return _u.uuid1(node, clock_seq)


def uuid3(namespace: UUID, name: str) -> UUID:
    return _u.uuid3(namespace, name)


def uuid4() -> UUID:
    return UUID(int=_uuid4_int())


def uuid5(namespace: UUID, name: str) -> UUID:
    return _u.uuid5(namespace, name)


def uuid6(node=None, timestamp: int | None = None) -> UUID:
    return UUID(str(_u.uuid6(node, timestamp)))


def uuid8(bytes_: bytes | None = None) -> UUID:
    return UUID(str(_u.uuid8(bytes_)))


__all__ = [
    "NIL", "MAX", "UUID", "SafeUUID", "getnode",
    "NAMESPACE_DNS", "NAMESPACE_OID", "NAMESPACE_URL", "NAMESPACE_X500",
    "RESERVED_FUTURE", "RESERVED_MICROSOFT", "RESERVED_NCS", "RFC_4122",
    "uuid1", "uuid3", "uuid4", "uuid5", "uuid6", "uuid7", "uuid8",
]
