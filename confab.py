"""Confab read-aloud client for Python agents.

Self-contained and dependency-free (standard library only, Python 3.8+). Copy this
file into an agent repository. The contract is documented in
docs/DEVELOPER_GUIDE.md of the Confab project.

    from confab import turn, speak
    t = turn(user_message, agent="my-agent", session_id=conversation_id)   # when the message arrives
    ...                                                                     # run the agent
    speak(final_answer, agent="my-agent", session_id=conversation_id, turn_id=t["turnId"])

The service speaks the reply only if that user message was sent with talk-back dictation
(there is no mode that speaks every reply). turn() and speak() never raise, never starts Confab, and never waits for speech. It
returns {"accepted": False, "reason": "unavailable"} when the service is not
running or does not answer within the timeout.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import uuid

DEFAULT_TIMEOUT = 0.25  # seconds; socket timeout for the connect and each read


def profile_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return os.path.join(base, "Confab", "resident")


def connection(profile: str | None = None) -> dict | None:
    """Read the port (endpoint.json) and bearer token (auth.json). None if absent."""
    profile = profile or profile_dir()
    try:
        with open(os.path.join(profile, "auth.json"), encoding="utf-8") as fh:
            token = json.load(fh).get("token")
        with open(os.path.join(profile, "endpoint.json"), encoding="utf-8") as fh:
            port = json.load(fh).get("port")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{64}", token):
        return None
    if not isinstance(port, int) or not 0 < port < 65536:
        return None
    return {"port": port, "token": token}


def _request(link: dict, method: str, route: str, body: dict | None, timeout: float) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", link["port"], timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        # http.client sends Host as 127.0.0.1:<port>, which the service requires, and
        # never sends Origin. Do not add an Origin or Sec-Fetch-* header.
        headers = {"Authorization": "Bearer " + link["token"]}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, route, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read(262145)
        value = json.loads(raw.decode("utf-8") or "{}") if len(raw) <= 262144 else {}
        if not isinstance(value, dict):
            value = {}
        value["status"] = response.status
        return value
    finally:
        conn.close()


def event(data: dict, *, agent: str = "custom", session_id: str = "default",
          profile: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """POST one event. Returns the service's {accepted, reason[, detail], status}."""
    link = connection(profile)
    if link is None:
        return {"accepted": False, "reason": "unavailable"}
    try:
        return _request(link, "POST", "/event", {**data, "agent": agent, "sessionId": session_id}, timeout)
    except Exception:  # timeouts, refused connections, malformed replies
        return {"accepted": False, "reason": "unavailable"}


def turn(text: str, *, agent: str = "custom", session_id: str = "default", turn_id: str | None = None,
         profile: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Report a user message as it reaches the agent. Returns the result plus "turnId".

    The service compares the text with the user's last talk-back dictation; the result's
    reason is "match" or "no_match". Pass the returned turnId to speak() for the reply.
    """
    turn_id = turn_id or str(uuid.uuid4())
    result = event({"event": "turn", "text": text, "turnId": turn_id},
                   agent=agent, session_id=session_id, profile=profile, timeout=timeout)
    return {**result, "turnId": turn_id}


def speak(text: str, *, agent: str = "custom", session_id: str = "default", id: str | None = None,
          turn_id: str | None = None, profile: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Send one final public reply. The service decides whether to speak it.

    Pass the turn_id from turn() so the reply is tied to its user message. Pass a stable id
    to make retries safe.
    """
    data = {"event": "reply", "text": text, "id": id or str(uuid.uuid4())}
    if turn_id:
        data["turnId"] = turn_id
    return event(data, agent=agent, session_id=session_id, profile=profile, timeout=timeout)


def status(*, profile: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """GET /health, or None when the service is unavailable."""
    link = connection(profile)
    if link is None:
        return None
    try:
        value = _request(link, "GET", "/health", None, timeout)
    except Exception:
        return None
    return value if value.pop("status", None) == 200 else None


if __name__ == "__main__":  # python confab.py "Sentence to speak"
    import sys
    print(json.dumps(speak(" ".join(sys.argv[1:]) or "Confab test.", agent="confab-cli")))
