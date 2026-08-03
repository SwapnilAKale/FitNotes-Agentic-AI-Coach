"""
The repo-root `uuid_utils` shim.

Windows Smart App Control (enforced) blocks unsigned low-reputation native
binaries. The installed uuid_utils ships an UNSIGNED .pyd, so importing it dies
with "An Application Control policy has blocked this file" — which took down
langchain_core -> langgraph -> the entire analytical pipeline on 2026-08-03.
Every other native extension in the venv loads fine; this one is niche.

The whole dependency chain uses exactly one function, `uuid7`, which is a short
spec. So the shim removes the native binary from the chain rather than
downgrading the machine's security.

These tests pin the contract, because a wrong UUID7 is the kind of thing that
looks fine and corrupts ordering silently.
"""

import importlib
import re
import time
from pathlib import Path
from uuid import UUID

import pytest

import uuid_utils
from uuid_utils import compat

_ROOT = Path(__file__).resolve().parent.parent


# ── it is our shim, and it is the one being imported ──────────────────────────

def test_the_repo_shim_is_what_gets_imported():
    assert Path(uuid_utils.__file__).resolve().is_relative_to(_ROOT), \
        "site-packages uuid_utils won: the repo root must precede it on sys.path"


def test_native_is_tried_first():
    """On a machine WITHOUT Smart App Control the real extension should still be
    used, so removing this shim later changes nothing."""
    assert hasattr(uuid_utils, "_NATIVE")
    if uuid_utils._NATIVE is None:
        assert uuid_utils._NATIVE_ERROR is not None, \
            "native module absent but no reason recorded"


# ── RFC 9562 section 5.7 ──────────────────────────────────────────────────────

def test_uuid7_is_a_real_uuid7():
    u = compat.uuid7()
    assert isinstance(u, UUID)
    assert u.version == 7
    assert (u.int >> 62) & 0b11 == 0b10, "variant bits must be 0b10"


def test_uuid7_timestamp_is_now():
    u = compat.uuid7()
    unix_ms = u.int >> 80
    assert abs(unix_ms - time.time() * 1000) < 5000


def test_uuid7_honours_an_explicit_timestamp():
    """An explicit timestamp is an instruction. The monotonicity guard used to
    drag it forward, so asking for 2023 silently returned 2026."""
    u = compat.uuid7(timestamp=1_700_000_000, nanos=500_000_000)
    unix_ms = u.int >> 80
    assert unix_ms == 1_700_000_000_500
    assert u.version == 7


def test_langchain_calls_both_forms():
    """langchain_core calls it bare AND with keywords — both must work."""
    assert compat.uuid7().version == 7
    assert compat.uuid7(timestamp=1, nanos=0).version == 7


def test_uuid7_is_monotonic_within_a_millisecond():
    """Time-ordering is the entire point; two UUIDs made in the same
    millisecond must still sort."""
    batch = [compat.uuid7() for _ in range(2000)]
    assert batch == sorted(batch, key=lambda u: u.int), "uuid7 is not monotonic"
    assert len(set(batch)) == len(batch), "uuid7 collided"


def test_uuid7_does_not_go_backwards_across_calls():
    a = compat.uuid7()
    b = compat.uuid7()
    assert b.int > a.int


# ── the surface the ecosystem actually imports ────────────────────────────────

def test_compat_exports_what_langchain_and_langsmith_import():
    for name in ("uuid7", "uuid1", "uuid3", "uuid4", "uuid5", "uuid6", "uuid8",
                 "NIL", "MAX", "UUID"):
        assert hasattr(compat, name), name
    from uuid_utils import _uuid4_int, _uuid7_int          # noqa: F401


@pytest.mark.parametrize("fn,ver", [("uuid4", 4), ("uuid6", 6), ("uuid7", 7),
                                    ("uuid8", 8)])
def test_versions_are_set(fn, ver):
    assert getattr(compat, fn)().version == ver


def test_named_uuids_match_the_stdlib():
    from uuid import NAMESPACE_DNS, uuid3 as s3, uuid5 as s5
    assert compat.uuid3(NAMESPACE_DNS, "x") == s3(NAMESPACE_DNS, "x")
    assert compat.uuid5(NAMESPACE_DNS, "x") == s5(NAMESPACE_DNS, "x")


def test_uuid8_rejects_a_wrong_length():
    with pytest.raises(ValueError):
        uuid_utils.uuid8(b"too short")


# ── the thing this was all for ────────────────────────────────────────────────

def test_langchain_core_uuid_module_imports():
    m = importlib.import_module("langchain_core.utils.uuid")
    assert m.uuid7().version == 7


def test_langgraph_imports():
    """The actual failure: langgraph -> langchain_core -> uuid_utils -> blocked
    .pyd, which broke src.graph.analytical and every test that touches it."""
    importlib.import_module("langgraph.graph")


def test_the_analytical_graph_imports():
    importlib.import_module("src.graph.analytical")


# ── documentation is part of the fix ──────────────────────────────────────────

def test_the_shim_explains_itself():
    src = (_ROOT / "uuid_utils" / "__init__.py").read_text(encoding="utf-8")
    for phrase in ("Smart App Control", "unsigned", "langchain_core", "RFC 9562"):
        assert phrase in src, f"the shim does not explain {phrase!r}"
