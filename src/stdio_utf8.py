import sys


def force_utf8_stdio(streams=None):
    """Reconfigure the given text streams (default: stdout + stderr) to UTF-8 so
    non-ASCII output (→ ← ✅ ⚠️ ✗ and emoji in reply text) can't raise
    UnicodeEncodeError on a CP1252 Windows console.

    Called at each directly-launched entry point (server.py, cli.py); the MCP
    subprocess is spawned with PYTHONIOENCODING=utf-8 instead. Streams without a
    .reconfigure() method (captured / redirected pipes) are skipped, and a
    reconfigure failure is swallowed so startup can never crash here. Returns the
    list of streams that were successfully reconfigured.
    """
    if streams is None:
        streams = [sys.stdout, sys.stderr]
    done = []
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
            done.append(stream)
        except Exception:
            pass
    return done
