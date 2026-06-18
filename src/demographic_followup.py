"""
src/demographic_followup.py
Stage B of the demographics feature — the conversational follow-up layer.

Pure, deterministic (NO LLM): detect when a user mentions a demographic in
passing, generate a gentle "want me to remember the precise anchor?" aside, and
interpret the user's immediate next reply into an anchor value to store via
Stage A's memory.set_demographic. The Coordinator owns the pending-state +
storage; this module owns the heuristics so they live in one testable home.

Design rules (from Stage B):
  - We never SCRAPE the mentioned value as the stored fact (a passing "22" is
    ambiguous) — the mention only TRIGGERS an offer to capture the precise
    anchor (birthdate, not "22"; training-start-date, not "3 years").
  - Conservative detection: prefer a miss over a false-fire. Bare numbers
    (reps/weights) must never trigger — an explicit demographic cue is required.

Parse conventions (stated):
  - birthdate: full ISO / "YYYY/MM/DD" / "Month [DD,] YYYY" (day→01 when absent);
    a BARE YEAR → clarify (age precision needs the month).
  - training_start_date: same, but a BARE YEAR → "YYYY-01-01" (accepted).
  - sex: mapped to the accepted set.
  - height: "176cm"→(176,"cm"); "5'9"/"5 ft 9"→ total inches (69,"in");
    a bare number → clarify the unit.
"""

import re
from typing import Optional

# Anchors we may proactively ask for (Stage-A STORABLE_KEYS), priority order.
ASKABLE = ("birthdate", "sex", "height", "training_start_date")

# ── Follow-up aside text (answer-first; this is appended on a new line) ───────
_FOLLOWUP_TEXT = {
    "birthdate": ("By the way — if you tell me your birthday I can factor your "
                  "exact age into answers like this going forward."),
    "training_start_date": ("By the way — if you tell me when you started "
                            "training, I can weigh your experience in answers "
                            "like this."),
    "height": ("By the way — if you share your height, I can tailor things like "
               "bodyweight-movement and proportion notes to you."),
    "sex": ("By the way — if you tell me whether you train as male or female, I "
            "can apply the right strength/bodyweight norms."),
}

ACK = "Got it — I'll remember that."

CLARIFY = {
    "birthdate": ("No rush — what's your date of birth? A full date like "
                  "1995-03-18 (or 'March 1995') works best."),
    "training_start_date": ("When did you start training? A year like 2021 is "
                            "fine, or a month and year."),
    "height": "What's your height? e.g. 176 cm or 5'9\".",
    "sex": "For norms — do you train as male or female?",
}


def followup_text(key: str) -> str:
    return _FOLLOWUP_TEXT.get(key, "")


# ── Mention detection (the trigger) ──────────────────────────────────────────
# Each pattern requires an explicit demographic cue so a bare number (reps,
# weight, sets) can never fire.
_AGE_RE = re.compile(r"\b\d{1,2}\s*[- ]?\s*(?:yo|y/o|years?[ -]old|yr[ -]old)\b"
                     r"|\bage[d]?\s+\d{1,2}\b", re.IGNORECASE)
_TENURE_RE = re.compile(
    r"\b(?:been\s+)?(?:train|lift|work(?:ing)?\s*out|in\s+the\s+gym)\w*\b[^.?!]{0,30}?"
    r"\b\d{1,2}\s*\+?\s*years?\b"
    r"|\b\d{1,2}\s*\+?\s*years?\b[^.?!]{0,20}?\b(?:of\s+)?(?:train|lift|gym|work(?:ing)?\s*out)\w*\b",
    re.IGNORECASE)
_HEIGHT_RE = re.compile(
    r"\b\d\s*'\s*\d{1,2}\s*\"?"                       # 5'9 / 5'9"
    r"|\b\d\s*(?:ft|foot|feet)\s*\d{1,2}\b"           # 5 ft 9
    r"|\b\d{2,3}\s*cm\b"                              # 176cm
    r"|\b(?:i'?m|i am|height\s+is|height)\s+\d{2,3}\s*cm\b",
    re.IGNORECASE)
_SEX_RE = re.compile(
    r"\b(?:as\s+a|i'?m\s+a?|i\s+am\s+a?)\s+(?:male|female|man|woman|guy|girl)\b"
    r"|\b(?:male|female)\s+here\b",
    re.IGNORECASE)


def detect_mentions(message: str) -> list:
    """Anchor keys the message references in passing (conservative). Ordered by
    ASKABLE priority. The VALUE is NOT extracted — the mention only triggers an
    offer to capture the precise anchor."""
    if not message:
        return []
    found = []
    if _AGE_RE.search(message):
        found.append("birthdate")
    if _SEX_RE.search(message):
        found.append("sex")
    if _HEIGHT_RE.search(message):
        found.append("height")
    if _TENURE_RE.search(message):
        found.append("training_start_date")
    # de-dup, preserve ASKABLE priority order
    return [k for k in ASKABLE if k in found]


# ── Answer interpretation (the user's immediate next reply) ──────────────────
_QUESTION_WORDS = ("how", "what", "why", "when", "where", "which", "who",
                   "should", "can", "could", "do", "does", "is", "are", "will",
                   "show", "tell", "give")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_SEX_MAP = {
    "male": "male", "m": "male", "man": "male", "guy": "male", "boy": "male",
    "female": "female", "f": "female", "woman": "female", "girl": "female",
    "intersex": "intersex", "other": "other",
}


def _looks_like_new_message(message: str) -> bool:
    """True if the message reads as a NEW question/request rather than a bare
    answer to the pending follow-up (→ drop the follow-up, route normally)."""
    m = (message or "").strip()
    if not m:
        return True
    if "?" in m:
        return True
    if len(m.split()) > 8:                       # a bare answer is short
        return True
    first = re.sub(r"[^a-z]", "", m.split()[0].lower())
    return first in _QUESTION_WORDS


def _answer(value, unit=None) -> dict:
    return {"kind": "answer", "value": value, "unit": unit}

_CLARIFY = {"kind": "clarify"}
_NOT_ANSWER = {"kind": "not_answer"}


def _parse_date(message: str, bare_year_mode: str):
    """
    Parse a date-ish answer → ('ok', iso) / ('clarify', None) / ('none', None).
    bare_year_mode: 'clarify' (birthdate) or 'jan1' (training_start_date).
    """
    m = message.strip()
    # full ISO or YYYY/MM/DD
    iso = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", m)
    if iso:
        y, mo, d = (int(g) for g in iso.groups())
        return _valid_ymd(y, mo, d)
    # Month name + year (+ optional day)
    mon = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})?,?\s*(\d{4})\b", m)
    if mon:
        name = mon.group(1)[:3].lower()
        if name in _MONTHS:
            y = int(mon.group(3)); mo = _MONTHS[name]
            d = int(mon.group(2)) if mon.group(2) else 1
            return _valid_ymd(y, mo, d)
    # "DD Month YYYY"
    dmy = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b", m)
    if dmy and dmy.group(2)[:3].lower() in _MONTHS:
        return _valid_ymd(int(dmy.group(3)), _MONTHS[dmy.group(2)[:3].lower()],
                          int(dmy.group(1)))
    # bare year
    yr = re.fullmatch(r"\D*?(\d{4})\D*?", m)
    if yr:
        y = int(yr.group(1))
        if bare_year_mode == "jan1":
            return _valid_ymd(y, 1, 1)
        return "clarify", None       # birthdate needs the month
    return "none", None


def _valid_ymd(y, mo, d):
    from datetime import date
    try:
        return "ok", date(y, mo, d).isoformat()
    except ValueError:
        return "clarify", None


def interpret_answer(key: str, message: str) -> dict:
    """
    Classify the user's next-turn reply to a pending follow-up for `key`:
      {"kind": "answer", "value", "unit"} | {"kind": "clarify"} | {"kind": "not_answer"}
    """
    if _looks_like_new_message(message):
        return dict(_NOT_ANSWER)
    m = (message or "").strip()

    if key in ("birthdate", "training_start_date"):
        mode = "jan1" if key == "training_start_date" else "clarify"
        status, iso = _parse_date(m, mode)
        if status == "ok":
            return _answer(iso)
        if status == "clarify":
            return dict(_CLARIFY)
        return dict(_NOT_ANSWER)

    if key == "sex":
        tok = re.sub(r"[^a-z]", "", m.lower())
        if tok in _SEX_MAP:
            return _answer(_SEX_MAP[tok])
        # a short reply that named a sex word anywhere
        for w in re.findall(r"[a-z]+", m.lower()):
            if w in _SEX_MAP:
                return _answer(_SEX_MAP[w])
        return dict(_NOT_ANSWER)

    if key == "height":
        # 5'9 / 5'9" / 5 ft 9  → inches
        fi = re.search(r"\b(\d)\s*(?:'|ft|foot|feet)\s*(\d{1,2})\b", m, re.IGNORECASE)
        if fi:
            inches = int(fi.group(1)) * 12 + int(fi.group(2))
            return _answer(float(inches), unit="in")
        # 176 cm / 176cm
        cm = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*cm\b", m, re.IGNORECASE)
        if cm:
            return _answer(float(cm.group(1)), unit="cm")
        inch = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*(?:in|inch|inches|\")\b", m, re.IGNORECASE)
        if inch:
            return _answer(float(inch.group(1)), unit="in")
        # a bare number → we have a value but no unit → clarify the unit
        if re.fullmatch(r"\D*?\d{2,3}(?:\.\d+)?\D*?", m):
            return dict(_CLARIFY)
        return dict(_NOT_ANSWER)

    return dict(_NOT_ANSWER)
