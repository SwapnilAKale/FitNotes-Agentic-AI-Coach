"""
src/units.py
Single source of truth for the kg-native rule — which exercises log their typed
weight in kilograms, and the SQL fragments that split a sum per unit frame.

Imported by (no circular risk — this module imports nothing internal):
  - src/data_agent/validate.py      (A2 unit checks: KG_NATIVE_EXERCISES, DEADLIFT_KG_SWITCH)
  - src/data_agent/process.py       (DEADLIFT_KG_SWITCH_DATE for the analytical predicate)
  - mcp_servers/combined_server.py  (per-unit volume CASE split + unit labelling)

The rule (unchanged from the three former copies):
  kg-native iff exercise_name in KG_NATIVE_NAMES  (kg for all history)
            OR  exercise_name == 'Deadlift' AND date >= DEADLIFT_KG_SWITCH.
"""

from datetime import date as _date, datetime as _datetime

# Exercises whose typed number is kilograms for ALL of history.
KG_NATIVE_NAMES: frozenset = frozenset({
    "Seated Machine Curl (Kg)",
    "Machine Wrist Extension",
    "Hand Gripper",
})

# Deadlift logged plates in lbs before this date, kg on/after.
DEADLIFT_KG_SWITCH = "2025-12-26"
DEADLIFT_KG_SWITCH_DATE = _datetime.strptime(DEADLIFT_KG_SWITCH, "%Y-%m-%d").date()

# Every exercise that may legitimately carry a kg unit (names + date-conditioned
# Deadlift). Membership form used by validate's A2 check.
KG_NATIVE_EXERCISES: frozenset = KG_NATIVE_NAMES | {"Deadlift"}


def is_kg_native(exercise_name: str, on_date=None) -> bool:
    """
    Canonical predicate: True iff the exercise's typed number is kilograms on
    the given date. The three KG_NATIVE_NAMES are kg for all history; Deadlift
    is kg only on/after DEADLIFT_KG_SWITCH. `on_date` may be a 'YYYY-MM-DD'
    string or a date; None means "current era" (Deadlift treated as kg).
    """
    if exercise_name in KG_NATIVE_NAMES:
        return True
    if exercise_name == "Deadlift":
        if on_date is None:
            return True
        d = on_date if isinstance(on_date, _date) else \
            _datetime.strptime(on_date, "%Y-%m-%d").date()
        return d >= DEADLIFT_KG_SWITCH_DATE
    return False


def kg_native_sql_predicate(name_col: str = "e.name", date_col: str = "tl.date") -> str:
    """
    SQL boolean that is TRUE for kg-native rows (date-aware for Deadlift).
    Built only from internal constants — safe to interpolate.
    """
    in_list = ", ".join("'" + n.replace("'", "''") + "'"
                        for n in sorted(KG_NATIVE_NAMES))
    return (f"({name_col} IN ({in_list}) "
            f"OR ({name_col} = 'Deadlift' AND {date_col} >= '{DEADLIFT_KG_SWITCH}'))")


def kg_native_volume_case(expr: str, name_col: str = "e.name",
                          date_col: str = "tl.date") -> tuple:
    """
    (lbs_sum_sql, kg_sum_sql) splitting a volume expression per unit frame so
    kilograms are never summed into pounds. The two buckets are different units
    and must never be added together.
    """
    pred = kg_native_sql_predicate(name_col, date_col)
    return (f"SUM(CASE WHEN NOT {pred} THEN {expr} ELSE 0 END)",
            f"SUM(CASE WHEN {pred} THEN {expr} ELSE 0 END)")
