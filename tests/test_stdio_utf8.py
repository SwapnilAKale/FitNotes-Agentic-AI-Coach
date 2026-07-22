import io

import pytest

from src.stdio_utf8 import force_utf8_stdio


def _cp1252_stream():
    """A text stream modelling a CP1252 Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")


def test_cp1252_stream_reconfigured_to_utf8():
    stream = _cp1252_stream()

    # Precondition: the arrow that crashes turn-start (residual #22) really does
    # raise on a CP1252 stream before the fix.
    with pytest.raises(UnicodeEncodeError):
        stream.write("→")  # →
        stream.flush()

    done = force_utf8_stdio([stream])

    assert stream in done
    assert stream.encoding == "utf-8"
    # The glyphs the app actually prints now encode without raising.
    stream.write("→ ✅ ⚠️")  # → ✅ ⚠️
    stream.flush()


def test_stream_without_reconfigure_is_skipped():
    # A bare bytes buffer has no .reconfigure(): must be skipped, not crash, and
    # must not appear in the reconfigured list.
    raw = io.BytesIO()
    assert not hasattr(raw, "reconfigure")

    done = force_utf8_stdio([raw])

    assert raw not in done
    assert done == []
