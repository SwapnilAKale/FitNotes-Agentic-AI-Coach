"""
src/demographics.py
Demographics tier model — the ONE source of truth for which user stats may be
stored, how they're validated, and how derived values are computed (Stage A of
the demographics feature). Pure logic, no I/O — so it's trivially testable with a
frozen `today` and there is exactly one place the rules live.

CORE PRINCIPLE (locked): store the ANCHOR, never the computed value. Birthdate
(not age), training-start-date (not years-trained). Derived values are computed
FRESH at point-of-use — the input is today's date, which changes daily, so a
stored derived value would go stale for zero perf gain. Same "one source of
truth, compute fresh" discipline as src/units.py.

THREE TIERS govern storage + use:
  TIER1 (use_freely)     — stable or accurately-derivable from the anchor, can
                           never be wrong: SEX (constant), BIRTHDATE → age,
                           TRAINING_START_DATE → years_trained.
  TIER2 (confirm_on_use) — known, changes slowly/monotonically: HEIGHT (only
                           rises/plateaus, never falls — the stored value is a
                           safe floor). Stored, but Stage B/C must confirm
                           ("has this changed since you told me?") before using
                           it in a personalized claim. Stage A only marks it.
  TIER3 (never_store)    — volatile any direction: BODYFAT %, BODYWEIGHT. NEVER
                           written to memory; Stage C asks fresh at point of use.

SENSITIVE-DATA STORAGE POLICY (lightweight discipline, not encryption):
  1. Only the storable demographic keys (TIER1/TIER2: birthdate, sex, height,
     training_start_date) may be stored as demographics — via
     memory.set_demographic (a structured map), NEVER as free-text facts.
  2. TIER3 (bodyfat_pct, bodyweight) are NEVER stored — asked fresh at use.
  3. Store the ANCHOR, never a computed/identity-derived value (no stored age).
  4. Demographics live in the structured `demographics` map and are NOT embedded
     in ChromaDB (kept out of the semantic vector store).

Stage B (follow-up asking) WRITES these via set_demographic; Stage C (RAG gate)
READS them via get_demographic / get_derived. Free-text extraction is locked OFF
demographics — they enter only through explicit follow-up answers.
"""

from datetime import date, datetime
from typing import Optional, Tuple

# ── Tiers ────────────────────────────────────────────────────────────────────
TIER1 = "use_freely"        # stable / accurately-derivable
TIER2 = "confirm_on_use"    # stored, but confirm before a personalized use
TIER3 = "never_store"       # volatile — ask fresh at point of use, never stored

# Accepted values for the `sex` constant (case-insensitive on input).
SEX_VALUES = frozenset({"male", "female", "intersex", "other"})

# Accepted height units.
HEIGHT_UNITS = frozenset({"cm", "in"})


# ── Per-stat registry (the single source of truth) ──────────────────────────
# Each entry: tier, is_anchor, derived (the derived key it computes or None),
# confirm_on_use. `stored` is False only for TIER3.
DEMOGRAPHIC_STATS = {
    "sex": {
        "tier": TIER1, "is_anchor": True, "derived": None,
        "confirm_on_use": False, "stored": True,
    },
    "birthdate": {
        "tier": TIER1, "is_anchor": True, "derived": "age",
        "confirm_on_use": False, "stored": True,
    },
    "training_start_date": {
        "tier": TIER1, "is_anchor": True, "derived": "years_trained",
        "confirm_on_use": False, "stored": True,
    },
    "height": {
        "tier": TIER2, "is_anchor": True, "derived": None,
        "confirm_on_use": True, "stored": True,
    },
    "bodyfat_pct": {
        "tier": TIER3, "is_anchor": False, "derived": None,
        "confirm_on_use": False, "stored": False,
    },
    "bodyweight": {
        "tier": TIER3, "is_anchor": False, "derived": None,
        "confirm_on_use": False, "stored": False,
    },
}

# Convenience sets derived from the registry (so callers never re-list keys).
STORABLE_KEYS = frozenset(k for k, v in DEMOGRAPHIC_STATS.items() if v["stored"])
TIER3_KEYS    = frozenset(k for k, v in DEMOGRAPHIC_STATS.items() if v["tier"] == TIER3)


# ── Validation ───────────────────────────────────────────────────────────────

def _parse_iso_date(value) -> Optional[date]:
    """Parse a strict YYYY-MM-DD date; None if not a real calendar date."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def validate_value(key: str, value, unit: Optional[str] = None,
                   today: Optional[date] = None) -> Tuple[bool, object, Optional[str]]:
    """
    Validate a demographic anchor for storage. Returns (ok, normalized, error).
    Does NOT decide storability by tier — that's the caller's policy gate; this
    only checks the value shape for an already-storable key.

    - birthdate / training_start_date: a real ISO date (YYYY-MM-DD), not in the future.
    - sex: one of SEX_VALUES (case-insensitive → normalized lower).
    - height: a positive number with unit in HEIGHT_UNITS.
    """
    today = today or date.today()

    if key in ("birthdate", "training_start_date"):
        d = _parse_iso_date(value)
        if d is None:
            return False, None, f"{key} must be a real date in YYYY-MM-DD form"
        if d > today:
            return False, None, f"{key} cannot be in the future"
        return True, d.isoformat(), None

    if key == "sex":
        if not isinstance(value, str):
            return False, None, "sex must be a string"
        norm = value.strip().lower()
        if norm not in SEX_VALUES:
            return False, None, f"sex must be one of {sorted(SEX_VALUES)}"
        return True, norm, None

    if key == "height":
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False, None, "height must be a number"
        if num <= 0:
            return False, None, "height must be positive"
        if unit not in HEIGHT_UNITS:
            return False, None, f"height unit must be one of {sorted(HEIGHT_UNITS)}"
        return True, num, None

    return False, None, f"{key} is not a validatable demographic anchor"


# ── Derived values (computed fresh, NEVER stored) ────────────────────────────

def _completed_years(anchor_iso: str, today: date) -> Optional[int]:
    d = _parse_iso_date(anchor_iso)
    if d is None or d > today:
        return None
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


def age_from_birthdate(birthdate_iso: str, today: Optional[date] = None) -> Optional[int]:
    """Full completed years between birthdate and today. Fresh every call."""
    return _completed_years(birthdate_iso, today or date.today())


def years_trained_from_start(start_iso: str, today: Optional[date] = None) -> Optional[int]:
    """Full completed years between training-start-date and today. Fresh every call."""
    return _completed_years(start_iso, today or date.today())


def compute_derived(key: str, anchor_value: str,
                    today: Optional[date] = None):
    """
    Compute the derived value for an anchor key, fresh, from today's date.
    Returns None if the key has no derived value. The result is NEVER persisted.
    """
    spec = DEMOGRAPHIC_STATS.get(key)
    if not spec or not spec.get("derived"):
        return None
    derived = spec["derived"]
    if derived == "age":
        return age_from_birthdate(anchor_value, today)
    if derived == "years_trained":
        return years_trained_from_start(anchor_value, today)
    return None
