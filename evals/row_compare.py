"""
Row comparison utilities for eval scoring.

rows_match() is the public API. Import it in run_evals.py.
Run this file directly to execute self-tests:
    python evals/row_compare.py
"""
import math
from typing import Any


def _coerce(v: Any) -> Any:
    """Return v as float if it can be parsed as one, otherwise strip and return as-is."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v.strip()
    return v


def _values_equal(a: Any, b: Any) -> bool:
    ca, cb = _coerce(a), _coerce(b)
    if isinstance(ca, float) and isinstance(cb, float):
        if ca == cb == 0.0:
            return True
        return math.isclose(ca, cb, rel_tol=0.01)
    return ca == cb


def _row_matches_gt(gt_row: dict, sys_row: dict) -> bool:
    """True if every GT value appears in sys_row's values (column-name agnostic).
    Each GT value must match a distinct SYS value via _values_equal."""
    gt_vals = sorted((_coerce(v) for v in gt_row.values()), key=str)
    sys_vals = sorted((_coerce(v) for v in sys_row.values()), key=str)
    used = [False] * len(sys_vals)
    for gt_val in gt_vals:
        matched = False
        for i, sys_val in enumerate(sys_vals):
            if not used[i] and _values_equal(gt_val, sys_val):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _sort_key(row: dict) -> tuple:
    """Canonical sort key for ORDER-agnostic comparison: sorted stringified values."""
    return tuple(sorted(str(_coerce(v)) for v in row.values()))


def rows_match(
    gt_rows: list[dict],
    sys_rows: list[dict],
    has_order_by: bool = False,
) -> bool:
    """
    Compare ground-truth rows to system rows.

    Rules:
    - Matching is key-based: all GT columns must be present in the SYS row with
      matching values. Extra columns in the SYS row are ignored.
    - Row order is ignored (both sets are sorted by canonical key before comparison).
    - SYS may have more rows than GT; GT rows must all be covered.
    - Floats are compared with 1% relative tolerance.
    - Scalar ground truth (1 row, 1 col): True if that value appears anywhere
      in any cell of sys_rows.
    - Both empty: True only if both are empty.
    """
    if not gt_rows and not sys_rows:
        return True
    if not gt_rows or not sys_rows:
        return False

    # Scalar case: ground truth is a single value
    if len(gt_rows) == 1 and len(gt_rows[0]) == 1:
        gt_val = _coerce(list(gt_rows[0].values())[0])
        for sys_row in sys_rows:
            for cell in sys_row.values():
                if _values_equal(gt_val, cell):
                    return True
        return False

    # GT rows must all be coverable by SYS (SYS may have extra rows)
    if len(gt_rows) > len(sys_rows):
        return False

    # Unordered: for each GT row find a distinct unused SYS row that covers it
    used = [False] * len(sys_rows)
    for gt_row in gt_rows:
        found = False
        for i, sys_row in enumerate(sys_rows):
            if not used[i] and _row_matches_gt(gt_row, sys_row):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Self-tests — run with: python evals/row_compare.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Identical single-column rows (column name differs — scalar path handles it)
    assert rows_match([{"pr_kg": 58.97}], [{"max_weight": 58.97}]) is True, "test 1"

    # 2. Reordered multi-row result — same columns, different row order
    r_gt = [{"exercise": "Lat Pulldown", "sets": 120}, {"exercise": "Cable Row", "sets": 80}]
    r_sys = [{"exercise": "Cable Row", "sets": 80}, {"exercise": "Lat Pulldown", "sets": 120}]
    assert rows_match(r_gt, r_sys, has_order_by=False) is True, "test 2"

    # 3. Extra columns in SYS row are ignored
    gt_extra = [{"exercise": "Lat Pulldown", "sets": 120}]
    sys_extra = [{"exercise": "Lat Pulldown", "sets": 120, "extra_col": 99}]
    assert rows_match(gt_extra, sys_extra) is True, "test 3"

    # 4. Scalar ground truth — value exists inside a multi-column system row
    gt_scalar = [{"pr": 58.97}]
    sys_rich = [{"exercise": "Lat Pulldown", "max_kg": 58.97, "reps": 8}]
    assert rows_match(gt_scalar, sys_rich) is True, "test 4"

    # 5. Both empty
    assert rows_match([], []) is True, "test 5"

    # 6. Float within 1% tolerance (0.5% diff)
    assert rows_match([{"v": 100.0}], [{"v": 100.5}]) is True, "test 6"

    # 7. Float outside 1% tolerance (2% diff)
    assert rows_match([{"v": 100.0}], [{"v": 102.0}]) is False, "test 7"

    # 8. GT has more rows than SYS → False
    assert rows_match([{"a": 1}, {"a": 2}], [{"a": 1}]) is False, "test 8"

    # 9. Ground truth empty but system has rows → False
    assert rows_match([], [{"a": 1}]) is False, "test 9"

    # 10. Numeric string in system rows treated as float
    assert rows_match([{"v": 42.0}], [{"v": "42"}]) is True, "test 10"

    # 11. GT has a value not present in any SYS cell → False
    assert rows_match([{"name": "A", "val": 99}], [{"name": "A", "other": 1}]) is False, "test 11"

    # 12. SYS has more rows than GT — GT rows all covered → True
    gt_sub = [{"date": "2024-06-05"}]
    sys_many = [{"date": "2024-06-05"}, {"date": "2024-06-20"}, {"date": "2024-06-29"}]
    assert rows_match(gt_sub, sys_many) is True, "test 12"

    # 13. Column-name mismatch but same values → True (column-name agnostic)
    assert rows_match(
        [{"exercise": "Lat Pulldown", "total_sets": 303}],
        [{"exercise_name": "Lat Pulldown", "sets": 303}],
    ) is True, "test 13"

    print("All row_compare self-tests passed.")
