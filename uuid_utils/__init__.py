"""
uuid_utils — compatibility shim.

WHY THIS EXISTS
Windows Smart App Control is ENFORCED on this machine. It blocks native
binaries that are both unsigned and low-reputation. The real `uuid_utils`
package ships `_uuid_utils.cp314-win_amd64.pyd`, which is **NotSigned**, so
loading it raises:

    ImportError: DLL load failed while importing _uuid_utils:
    An Application Control policy has blocked this file.

That took down `langchain_core.utils.uuid` -> `langgraph` -> the entire
analytical pipeline, and every test that imports it (2026-08-03). Every other
native extension in the venv (numpy, torch, chromadb, xxhash, regex) loads
fine, because those are high-reputation; this one is a niche package.

Reinstalling does not help — you get the same unsigned binary. Disabling Smart
App Control is a system-wide, irreversible security downgrade to accommodate one
file. So the root fix is to remove the need for the binary at all.

WHAT IT IS FOR
The whole dependency chain uses exactly ONE function:

    langchain_core/utils/uuid.py:12   from uuid_utils.compat import uuid7
    langsmith/_internal/_uuid.py:11   from uuid_utils.compat import uuid7

A 351 KB Rust extension, to concatenate a timestamp with random bits. UUIDv7 is
a short spec (RFC 9562 §5.7) and needs no native code.

HOW IT BEHAVES
Native-first: it tries to load the real compiled module and re-exports it
unchanged when the OS permits. Only when that is blocked does it fall back to
the pure-Python implementation below. So this is correct on a machine WITHOUT
Smart App Control too — nothing is downgraded, and removing the shim later
changes no behaviour.

This package sits at the repo root, which precedes site-packages on sys.path,
so it shadows the installed one. It is version-controlled, so a pip reinstall
cannot silently reintroduce the failure, and tests/test_uuid_utils_shim.py
fails loudly if the contract ever drifts.
"""

from __future__ import annotations

import importlib.util
import os
import secrets
import sys
import time
import uuid as _uuid
from uuid import UUID

__version__ = "0.16.2+shim"

# ── Native-first ──────────────────────────────────────────────────────────────
# Load the real extension if the OS allows it. Anything it exports wins; the
# pure-Python definitions below are only a fallback.
_NATIVE = None
_NATIVE_ERROR = None


def _load_native():
    """Import the compiled _uuid_utils from the INSTALLED package, bypassing
    this shim. Returns the module, or None when the OS blocks it."""
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))):
            continue                       # skip the repo root (this shim)
        pkg = os.path.join(entry, "uuid_utils")
        if not os.path.isdir(pkg):
            continue
        for name in sorted(os.listdir(pkg)):
            if name.startswith("_uuid_utils") and name.endswith((".pyd", ".so")):
                spec = importlib.util.spec_from_file_location(
                    "_uuid_utils_native", os.path.join(pkg, name))
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)     # raises if blocked
                return mod
    return None


try:
    _NATIVE = _load_native()
except Exception as exc:                      # blocked, missing, or ABI mismatch
    _NATIVE_ERROR = exc

# ── Pure-Python fallback ──────────────────────────────────────────────────────

_UUID7_LAST = (0, 0)     # (unix_ms, counter) — monotonicity within a millisecond


def _uuid7_int(timestamp: int | None = None, nanos: int = 0) -> int:
    """
    RFC 9562 section 5.7 layout:

        48 bits  unix_ts_ms
         4 bits  version (7)
        12 bits  rand_a      — used here as a sub-millisecond counter so UUIDs
                               generated in the same millisecond stay ordered
         2 bits  variant (0b10)
        62 bits  rand_b
    """
    global _UUID7_LAST

    if timestamp is not None:
        # An EXPLICIT timestamp is an instruction, not a suggestion. The
        # monotonicity guard below must not drag it forward to "now" — doing so
        # silently returned a 2026 UUID when 2023 was asked for.
        unix_ms = timestamp * 1_000 + nanos // 1_000_000
        value = (unix_ms & 0xFFFF_FFFF_FFFF) << 80
        value |= 0x7 << 76
        value |= secrets.randbits(12) << 64
        value |= 0b10 << 62
        value |= secrets.randbits(62)
        return value

    unix_ms = time.time_ns() // 1_000_000
    last_ms, last_seq = _UUID7_LAST
    if unix_ms == last_ms:
        seq = last_seq + 1
        if seq > 0xFFF:                    # counter exhausted; step the clock
            unix_ms += 1
            seq = 0
    elif unix_ms < last_ms:
        unix_ms, seq = last_ms, last_seq + 1   # clock went backwards; don't
        if seq > 0xFFF:
            unix_ms += 1
            seq = 0
    else:
        seq = secrets.randbits(12) >> 2    # start low, leave counter headroom
    _UUID7_LAST = (unix_ms, seq)

    value = (unix_ms & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x7 << 76
    value |= (seq & 0xFFF) << 64
    value |= 0b10 << 62
    value |= secrets.randbits(62)
    return value


def _uuid4_int() -> int:
    return _uuid.uuid4().int


def uuid7(timestamp: int | None = None, nanos: int = 0) -> UUID:
    """Time-ordered UUID. Monotonic within a millisecond."""
    return UUID(int=_uuid7_int(timestamp, nanos))


def uuid1(node=None, clock_seq=None) -> UUID:
    return _uuid.uuid1(node, clock_seq)


def uuid3(namespace: UUID, name: str) -> UUID:
    return _uuid.uuid3(namespace, name)


def uuid4() -> UUID:
    return _uuid.uuid4()


def uuid5(namespace: UUID, name: str) -> UUID:
    return _uuid.uuid5(namespace, name)


def uuid6(node=None, timestamp: int | None = None) -> UUID:
    """RFC 9562 section 5.6 — uuid1's fields reordered so it sorts by time."""
    u = _uuid.uuid1(node)
    t = u.time                                  # 60-bit gregorian
    value = ((t >> 12) & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x6 << 76
    value |= (t & 0xFFF) << 64
    value |= 0b10 << 62
    value |= u.int & ((1 << 62) - 1)
    return UUID(int=value)


def uuid8(bytes_: bytes | None = None) -> UUID:
    """RFC 9562 section 5.8 — custom/vendor-defined, version and variant fixed."""
    raw = bytes_ if bytes_ is not None else secrets.token_bytes(16)
    if len(raw) != 16:
        raise ValueError("uuid8 requires exactly 16 bytes")
    value = int.from_bytes(raw, "big")
    value &= ~(0xF << 76)
    value |= 0x8 << 76
    value &= ~(0b11 << 62)
    value |= 0b10 << 62
    return UUID(int=value)


# Re-export the native implementations when they are available, so a machine
# without Smart App Control behaves exactly as it did before this shim existed.
if _NATIVE is not None:                        # pragma: no cover - env-dependent
    for _name in ("uuid1", "uuid3", "uuid4", "uuid5", "uuid6", "uuid7", "uuid8",
                  "_uuid4_int", "_uuid7_int"):
        if hasattr(_NATIVE, _name):
            globals()[_name] = getattr(_NATIVE, _name)
    __version__ = getattr(_NATIVE, "__version__", __version__)

__all__ = ["UUID", "uuid1", "uuid3", "uuid4", "uuid5", "uuid6", "uuid7", "uuid8",
           "_uuid4_int", "_uuid7_int", "__version__"]
