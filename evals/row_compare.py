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


def _row_to_values(row: dict) -> tuple:
    """Extract row values as a tuple in dict-insertion order (column names ignored)."""
    return tuple(_coerce(v) for v in row.values())


def _value_lists_match(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    return all(_values_equal(av, bv) for av, bv in zip(a, b))


def rows_match(
    gt_rows: list[dict],
    sys_rows: list[dict],
    has_order_by: bool = False,
) -> bool:
    """
    Compare ground-truth rows to system rows.

    Rules:
    - Column names are ignored; values are compared by position within each row.
    - Row order is ignored unless has_order_by is True.
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

    # Row counts must match for non-scalar cases
    if len(gt_rows) != len(sys_rows):
        return False

    gt_tuples = [_row_to_values(r) for r in gt_rows]
    sys_tuples = [_row_to_values(r) for r in sys_rows]

    if has_order_by:
        return all(_value_lists_match(g, s) for g, s in zip(gt_tuples, sys_tuples))

    # Unordered: match each GT row to exactly one SYS row
    used = [False] * len(sys_tuples)
    for gt_t in gt_tuples:
        found = False
        for i, sys_t in enumerate(sys_tuples):
            if not used[i] and _value_lists_match(gt_t, sys_t):
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
    # 1. Identical single-column rows (column name differs)
    assert rows_match([{"pr_kg": 58.97}], [{"max_weight": 58.97}]) is True, "test 1"

    # 2. Reordered multi-row result without ORDER BY
    r_gt = [{"name": "Lat Pulldown", "sets": 120}, {"name": "Cable Row", "sets": 80}]
    r_sys = [{"exercise": "Cable Row", "cnt": 80}, {"exercise": "Lat Pulldown", "cnt": 120}]
    assert rows_match(r_gt, r_sys, has_order_by=False) is True, "test 2"

    # 3. Same rows but wrong order WITH ORDER BY
    assert rows_match(r_gt, r_sys, has_order_by=True) is False, "test 3"

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

    # 8. Different row counts → False
    assert rows_match([{"a": 1}, {"a": 2}], [{"a": 1}]) is False, "test 8"

    # 9. Ground truth empty but system has rows → False
    assert rows_match([], [{"a": 1}]) is False, "test 9"

    # 10. Numeric string in system rows treated as float
    assert rows_match([{"v": 42.0}], [{"v": "42"}]) is True, "test 10"

    print("All row_compare self-tests passed.")
