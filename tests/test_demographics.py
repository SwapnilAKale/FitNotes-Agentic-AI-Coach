"""
Demographics feature — Stage A (storage layer).

Tier model (one source of truth, src/demographics.py) + anchor storage +
compute-on-demand derived values + the sensitive-data storage policy. Locked
principle: store the ANCHOR (birthdate, start-date), never the computed value
(age, years-trained) — derived fresh at point-of-use.

No Gemini. The store is isolated to a tmp file (never touches data/memory.json);
demographics don't touch ChromaDB, so no Chroma here.
"""

import json
import os
import sys
from datetime import date

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import demographics as D            # noqa: E402
from src import memory                       # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Isolate the memory store to a tmp file."""
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "mem.json")
    return tmp_path / "mem.json"


def _raw_demographics(path):
    return json.loads(path.read_text())["demographics"] if path.exists() else {}


# ── Tier model ───────────────────────────────────────────────────────────────

def test_tier_model_mapping():
    s = D.DEMOGRAPHIC_STATS
    assert s["sex"]["tier"] == D.TIER1
    assert s["birthdate"]["tier"] == D.TIER1 and s["birthdate"]["derived"] == "age"
    assert s["training_start_date"]["tier"] == D.TIER1
    assert s["training_start_date"]["derived"] == "years_trained"
    assert s["height"]["tier"] == D.TIER2 and s["height"]["confirm_on_use"] is True
    assert s["bodyfat_pct"]["tier"] == D.TIER3 and s["bodyfat_pct"]["stored"] is False
    assert s["bodyweight"]["tier"] == D.TIER3 and s["bodyweight"]["stored"] is False


def test_storable_and_tier3_key_sets():
    assert set(D.STORABLE_KEYS) == {"birthdate", "sex", "height", "training_start_date"}
    assert set(D.TIER3_KEYS) == {"bodyfat_pct", "bodyweight"}


# ── set_demographic: valid stores ────────────────────────────────────────────

def test_set_valid_anchors_store_and_read_back(store):
    assert memory.set_demographic("birthdate", "2003-01-15")["status"] == "saved"
    assert memory.set_demographic("sex", "Male")["status"] == "saved"       # normalized
    assert memory.set_demographic("height", 176, unit="cm")["status"] == "saved"
    assert memory.set_demographic("training_start_date", "2023-06-01")["status"] == "saved"

    assert memory.get_demographic("birthdate")["value"] == "2003-01-15"
    assert memory.get_demographic("sex")["value"] == "male"                 # case-normalized
    h = memory.get_demographic("height")
    assert h["value"] == 176 and h["unit"] == "cm" and h["tier"] == D.TIER2
    assert memory.get_demographic("nonexistent") is None


def test_set_overwrites_anchor_correction(store):
    memory.set_demographic("birthdate", "2003-01-15")
    memory.set_demographic("birthdate", "2003-02-20")   # correction
    assert memory.get_demographic("birthdate")["value"] == "2003-02-20"
    assert len(_raw_demographics(store)) == 1            # single-valued per key


# ── set_demographic: tier-3 refusal (sensitive-data policy) ─────────────────

@pytest.mark.parametrize("key,val", [("bodyfat_pct", 12.0), ("bodyweight", 80.0)])
def test_tier3_never_stored(store, key, val):
    out = memory.set_demographic(key, val)
    assert out["status"] == "refused"
    assert "tier-3" in out["reason"] and "never stored" in out["reason"]
    assert key not in _raw_demographics(store)           # nothing written


def test_unknown_key_refused(store):
    assert memory.set_demographic("favorite_color", "blue")["status"] == "refused"
    assert _raw_demographics(store) == {}


# ── set_demographic: malformed values rejected ──────────────────────────────

@pytest.mark.parametrize("key,val,unit", [
    ("birthdate", "2003-13-99", None),     # impossible date
    ("birthdate", "2999-01-01", None),     # future
    ("training_start_date", "not-a-date", None),
    ("height", 0, "cm"),                   # non-positive
    ("height", -5, "cm"),
    ("height", 176, "furlongs"),           # bad unit
    ("sex", "alien", None),                # outside the set
])
def test_malformed_values_rejected(store, key, val, unit):
    out = memory.set_demographic(key, val, unit=unit)
    assert out["status"] == "invalid" and out["reason"]
    assert _raw_demographics(store) == {}   # nothing written on a rejected value


# ── get_derived: computed FRESH, never stored ───────────────────────────────

def test_age_computed_fresh_changes_with_today(store):
    memory.set_demographic("birthdate", "2003-06-18")
    assert memory.get_derived("birthdate", today=date(2025, 6, 18)) == 22
    # advancing 'today' a year recomputes — proves it's not a cached/stored value
    assert memory.get_derived("birthdate", today=date(2026, 6, 18)) == 23
    # day before the 2026 birthday → still 22 (completed-years correctness)
    assert memory.get_derived("birthdate", today=date(2026, 6, 17)) == 22


def test_years_trained_computed_fresh(store):
    memory.set_demographic("training_start_date", "2023-06-01")
    assert memory.get_derived("training_start_date", today=date(2025, 6, 18)) == 2
    assert memory.get_derived("training_start_date", today=date(2026, 6, 18)) == 3


def test_no_derived_value_is_ever_persisted(store):
    memory.set_demographic("birthdate", "2003-06-18")
    memory.set_demographic("training_start_date", "2023-06-01")
    memory.get_derived("birthdate", today=date(2025, 6, 18))
    memory.get_derived("training_start_date", today=date(2025, 6, 18))
    stored = _raw_demographics(store)
    assert set(stored) == {"birthdate", "training_start_date"}   # anchors only
    assert "age" not in stored and "years_trained" not in stored


def test_get_derived_none_when_unstored_or_no_derived(store):
    assert memory.get_derived("birthdate") is None        # not stored yet
    memory.set_demographic("height", 176, unit="cm")
    assert memory.get_derived("height", today=date(2025, 6, 18)) is None  # no derived


# ── pure compute helpers (today-injectable) ─────────────────────────────────

def test_pure_compute_helpers():
    assert D.age_from_birthdate("2000-01-01", today=date(2025, 1, 1)) == 25
    assert D.age_from_birthdate("2000-01-02", today=date(2025, 1, 1)) == 24  # bday not reached
    assert D.years_trained_from_start("2020-03-15", today=date(2025, 3, 14)) == 4
    assert D.compute_derived("sex", "male", today=date(2025, 1, 1)) is None  # no derived


# ── demographics stay out of the free-text facts / ChromaDB path ────────────

def test_demographics_not_stored_as_freetext_facts(store):
    memory.set_demographic("birthdate", "2003-01-15")
    assert memory.get_all_facts() == []     # demographics live in their own map
