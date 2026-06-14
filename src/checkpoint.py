"""
src/checkpoint.py
Single-slot checkpoint for quota-interrupted questions.

When a rate-limit 429 kills a question mid-pipeline, the pipeline saves the
minimal state needed to resume without re-paying for completed LLM stages:

  - The analytical package is NEVER stored — it rebuilds free (pure Python).
  - The draft answer IS stored VERBATIM — never summarized, because the
    grounding check must verify the exact text the user will see.
  - POLICY: the user never sees unverified draft text. On a grounding
    interruption the user gets a status message only; the answer ships
    solely after grounding completes on resume.

Slot file: data/checkpoint.json (gitignored). Latest interrupted question
only — a new question while a checkpoint exists discards it.
"""

import json
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

MAX_AGE_HOURS         = 48
PRUNE_TOOL_RESULT_AT  = 2000   # chars; tool results above this are head+tail pruned
PRUNE_MARKER          = "[truncated for resume]"

# Optional google error types — same detection logic as coordinator._is_rate_limit
try:
    from google.genai import errors as _genai_errors
    _GenaiClientError = _genai_errors.ClientError
except (ImportError, AttributeError):
    _GenaiClientError = None  # type: ignore[assignment]

try:
    from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
except ImportError:
    _ResourceExhausted = None  # type: ignore[assignment]


def is_rate_limit(exc: Exception) -> bool:
    """True if exc is a Gemini/google-api 429 / ResourceExhausted error."""
    if _ResourceExhausted is not None and isinstance(exc, _ResourceExhausted):
        return True
    if _GenaiClientError is not None and isinstance(exc, _GenaiClientError):
        if getattr(exc, "code", None) == 429:
            return True
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


class QuotaInterrupted(Exception):
    """
    Raised AFTER a 429 has been checkpoint-saved.

    str() preserves the original error text (so the server's
    retry_after_seconds parser and every "429" string match keep working);
    `user_message` carries the status line to show the user — it never
    contains draft content.
    """
    def __init__(self, original: Exception, user_message: str):
        self.original         = original
        self.user_message     = user_message
        self.checkpoint_saved = True
        msg = str(original)
        if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
            msg = f"429 RESOURCE_EXHAUSTED: {msg}"
        super().__init__(msg)


# ── User-facing status messages (no draft content, ever) ──────────────────────

MSG_VERIFY_INTERRUPTED = (
    "Analysis drafted, but verification was interrupted by the quota limit. "
    "Say 'continue' when quota resets — the answer will be shown once verified."
)

MSG_OPERATIONAL_INTERRUPTED = (
    "Your request was interrupted by the quota limit mid-task. "
    "Say 'continue' when quota resets — completed steps were saved and "
    "won't be repeated."
)


def _retry_seconds(exc: Exception) -> int | None:
    """Best-effort retry-delay extraction for the countdown in status text."""
    msg = str(exc).lower()
    m = re.search(r"retrydelay['\"]?\s*:\s*['\"]?(\d+\.?\d*)s", msg)
    if not m:
        m = re.search(r"in (\d+\.?\d*)s", msg)
    return int(float(m.group(1))) if m else None


def msg_draft_interrupted(exc: Exception) -> str:
    secs = _retry_seconds(exc)
    countdown = f" (countdown: {secs}s)" if secs is not None else ""
    return (
        f"Quota hit before the analysis could run. Say 'continue' when quota "
        f"resets{countdown} — your question is saved."
    )


# ── Continue-intent detection ─────────────────────────────────────────────────

_CONTINUE_RE = re.compile(
    r"(?i)^\s*(?:please\s+|ok,?\s+|yes,?\s+)?"
    r"(?:continue|resume|carry\s*on)"
    r"(?:\s+(?:please|now))?[\s!.…]*$"
)


def is_continue_intent(message: str) -> bool:
    """Short, bare continue/resume/carry-on message (case-insensitive)."""
    return bool(message) and len(message) <= 40 and bool(_CONTINUE_RE.match(message))


_DISCARD_RE = re.compile(
    r"(?i)^\s*(?:new|discard|start\s+over|start\s+fresh|fresh\s+start|fresh|"
    r"scrap\s+it|forget\s+it|never\s*mind|nevermind)[\s!.…]*$"
)

# Bare confirmations / fillers that answer NEITHER 'continue' nor 'new' clearly.
_AMBIGUOUS_FILLERS = frozenset({
    "ok", "okay", "k", "sure", "yes", "yeah", "yep", "yup", "ya",
    "no", "nope", "nah", "hmm", "hm", "what", "huh", "idk", "dunno",
    "maybe", "wait", "um", "uh", "well", "so",
})


def is_discard_intent(message: str) -> bool:
    """Explicit 'discard the saved question and start fresh' reply."""
    return bool(message) and bool(_DISCARD_RE.match(message))


def is_ambiguous_reply(message: str) -> bool:
    """
    A reply that resolves the discard prompt neither way — a bare filler or
    confirmation. Used only while awaiting_discard_confirm; on these we re-ask
    rather than guess (never silently discard).
    """
    if not message:
        return True
    return message.strip().lower().rstrip("?!.… ") in _AMBIGUOUS_FILLERS


def discard_confirm_prompt(cp: dict) -> str:
    """
    Prompt shown when a NEW question arrives while a live slot exists. Names
    the saved question and its verification state. No draft content.
    """
    saved = cp.get("question") or "(your previous question)"
    route = cp.get("route")
    stage = cp.get("completed_stage")
    if route == "operational":
        state = "in progress"
    elif stage in ("draft", "coverage"):
        state = "drafted, verification pending"
    else:
        state = "saved, not yet started"
    return (
        f'You have an interrupted question waiting: "{saved}" ({state}). '
        f"Do you want to:\n"
        f"  • say 'continue' to finish it, or\n"
        f"  • say 'new' (or repeat your question) to discard it and start fresh?"
    )


# ── Slot I/O ──────────────────────────────────────────────────────────────────

def _path() -> str:
    return os.environ.get("CHECKPOINT_PATH", os.path.join("data", "checkpoint.json"))


def prune_tool_messages(messages: list, limit: int = PRUNE_TOOL_RESULT_AT) -> list:
    """
    MECHANICAL pruning for operational checkpoints: any tool-result content
    over `limit` chars becomes head + marker + tail. No LLM summarization —
    exact values (ids, weights, dates) in the kept portions are untouched.
    """
    pruned = []
    keep = limit // 2
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if (m.get("role") == "tool" and isinstance(content, str)
                and len(content) > limit):
            m["content"] = (content[:keep] + f"\n{PRUNE_MARKER}\n" + content[-keep:])
        pruned.append(m)
    return pruned


def save_checkpoint(route:           str,
                    question:        str,
                    params:          dict | None = None,
                    completed_stage: str | None = None,
                    draft:           str | None = None,
                    messages:        list | None = None) -> dict:
    """
    Write the single slot, overwriting any previous checkpoint.
    The draft is stored VERBATIM. Operational messages are mechanically pruned.
    """
    cp = {
        "created":         datetime.now().isoformat(),
        "route":           route,
        "question":        question,
        "params":          params,
        "completed_stage": completed_stage,
        "draft":           draft,
        "messages":        prune_tool_messages(messages) if messages else None,
    }
    _write(cp)
    logger.info(
        "[checkpoint] saved: route=%s completed_stage=%s draft=%s messages=%s",
        route, completed_stage,
        f"{len(draft)} chars" if draft else "none",
        len(messages) if messages else 0,
    )
    return cp


def _write(cp: dict) -> None:
    path = _path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cp, f, default=str)


def mark_awaiting_discard(cp: dict, pending_question: str) -> dict:
    """
    Flag the live slot as awaiting a discard-confirmation reply, stashing the
    new question that triggered the prompt. The NEXT message is then
    interpreted as the answer to the prompt. Idempotent re-prompts keep the
    original pending_question.
    """
    cp = dict(cp)
    cp["awaiting_discard_confirm"] = True
    cp.setdefault("pending_question", pending_question)
    _write(cp)
    return cp


def load_checkpoint() -> dict | None:
    """
    Load the slot. Returns None when missing, corrupt, or older than
    MAX_AGE_HOURS (stale/corrupt slots are discarded with a log line).
    """
    path = _path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cp = json.load(f)
        created = datetime.fromisoformat(cp["created"])
        age_h = (datetime.now() - created).total_seconds() / 3600
        if age_h > MAX_AGE_HOURS:
            logger.info("[checkpoint] discarded stale slot (%.0fh old)", age_h)
            clear_checkpoint()
            return None
        return cp
    except Exception as e:
        logger.warning("[checkpoint] corrupt slot discarded: %s", e)
        clear_checkpoint()
        return None


def clear_checkpoint() -> None:
    try:
        os.remove(_path())
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("[checkpoint] could not remove slot: %s", e)
