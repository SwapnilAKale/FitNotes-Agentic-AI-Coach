"""
src/data_agent/session_display.py
Deterministic session-display for the Data Agent (analytical lane).

STAGE 1 of moving session-display off the operational agent. This module is
built and unit-tested in ISOLATION — it is wired to nobody yet.

Two modes, ONE shared display core (_build_session_displays):

  get_exercise_sessions(exercise_name, mode, ...)  single-exercise display
  get_category_session(category, target)           NEW category-level display

DUPLICATION IS INTENTIONAL (this stage only). The display orchestration
(per-date grouping, drop-set grouping, warmup labeling, verbatim display_sets
assembly, BAR_EXERCISE_NOTES, the _bar_inclusive_weight wrapper) is copied
VERBATIM from mcp_servers/combined_server.py:_get_exercise_sessions_sync
(~:2172-2349). The operational copy is scheduled for deletion in a later stage;
extracting/rewiring it now would refactor doomed code and muddy the
new-vs-operational equality test (tests/test_session_display.py::T3). The ONLY
logic shared with the operational path is the already-shared leaf math in
src/data_agent/process.py (bar / offset / kg-native / typed-weight recovery),
imported below — never re-copied.

Single-exercise output is byte-identical to the operational get_exercise_sessions
(proven by the equality test). The only genuinely new logic is category mode.
"""

import re

from src.db import get_connection
from src.data_agent.fetch import (
    load_user_context, DB_PATH, EXCLUDED_CATEGORY_IDS,
)
# Shared leaf primitives — the SINGLE source of truth for the headline weight
# math, identical to the analytical package, get_weekly_volume, and the
# operational display reads. Reused, never re-copied.
from src.data_agent.process import (
    _get_bar_weight_lbs   as _proc_bar_weight_lbs,
    _get_numeric_offset   as _proc_numeric_offset,
    _is_kg_native         as _proc_is_kg_native,
    _recover_typed_weight as _proc_recover_typed_weight,
    CATEGORY_NAMES,
    match_muscle_group,
)


# ── BAR_EXERCISE_NOTES (verbatim port of combined_server.py:740-748) ─────────────
BAR_EXERCISE_NOTES = {
    "Deadlift": "Weights shown are plate weights only (kg). Add 20kg bar for total weight.",
    "Barbell Row": "Weights shown are plate weights only (lbs). Add 44.09 lbs bar for total weight.",
    "Barbell Curl": "Weights shown are plate weights only (lbs). Bar weight varies by date — see user_context.",
    "Barbell Upright Row": "Weights shown are plate weights only (lbs). Bar weight varies by date — see user_context.",
    "Behind The Back Wrist Curls": "Weights shown are plate weights only (lbs). Bar weight varies by date — see user_context.",
    "EZ-Bar Curl": "Weights shown are plate weights only (lbs). Add 22.05 lbs bar for total weight.",
    "Reverse Zig Zag Barbell Curls": "Weights shown are plate weights only (lbs). Add 22.05 lbs bar for total weight.",
}


# ── _bar_inclusive_weight (verbatim port of combined_server.py:767-790) ──────────
# Composes the imported process.py primitives — same math as the analytical
# package's per-set headline. Smith counterbalance reductions and per-set
# comment-unit overrides (analytical-only refinements) are NOT applied here, same
# as the operational display reads.
def _bar_inclusive_weight(ctx: dict, exercise_name: str, date_str: str,
                          metric_weight: float):
    """
    Convert a raw training_log.metric_weight to the BAR-INCLUSIVE headline weight
    for a given exercise/date, plus its unit label and the plates-only value.

    Returns (headline_weight, unit, plates) where:
      plates   = metric_weight * 2.2046 + numeric_offset  (recovers the logged number)
      bar      = date-ranged bar weight, in the set's own frame (kg-native -> kg)
      headline = round(plates + bar, 1)
      unit     = "kg" for kg-native exercises (date-ranged for Deadlift), else "lbs"
    """
    offset  = _proc_numeric_offset(ctx, exercise_name)
    plates  = _proc_recover_typed_weight(metric_weight, offset)
    is_kg   = _proc_is_kg_native(ctx, exercise_name, date_str)
    bar_lbs = _proc_bar_weight_lbs(ctx, exercise_name, date_str)
    bar     = bar_lbs / 2.2046 if is_kg else bar_lbs
    return round(plates + bar, 1), ("kg" if is_kg else "lbs"), plates


# ── Drop-set label parsing (verbatim port of combined_server.py:2282-2292) ───────
_SET_LABEL_RE = re.compile(r'\b(\d+)(?:st|nd|rd|th)\s+set\b', re.IGNORECASE)


def _parse_set_num(text):
    m = _SET_LABEL_RE.search(text or "")
    return int(m.group(1)) if m else None


def _strip_set_label(text):
    if not text:
        return None
    stripped = _SET_LABEL_RE.sub('', text).strip(' ,.-')
    return stripped or None


# ── display_sets assembly (verbatim port of combined_server.py:2294-2333) ────────
def _cardio_set_line(s) -> str:
    """Cardio per-set content: distance/duration/pace, never weight×reps.
    Pace uses the SAME formula + rounding as the package's pace_min_per_km
    (process.py `_build_session_dict`), so display and PR-layer pace never disagree.
    """
    dist = s.get("distance") or 0
    dur  = s.get("duration_seconds") or 0
    if dist > 0:
        line = f"{dist} km in {dur}s"
        if dur > 0:
            pace = round((dur / 60) / dist, 2)
            line += f" ({pace} min/km)"
    else:
        line = f"{dur}s"
    if s.get("comment"):
        line += f" ({s['comment'].strip()})"
    return line


def _build_display_sets(sess_sets, unit, is_cardio=False):
    if not sess_sets:
        return []
    if is_cardio:
        # Cardio has no weight/reps, warmups, or drop-sets — render distance/
        # duration/pace directly, one "Set N" line per logged entry.
        return [f"Set {i}: {_cardio_set_line(s)}" for i, s in enumerate(sess_sets, 1)]
    # Display the bar-inclusive weight, but compute the warmup ratio on
    # PLATES (pre-bar) so the constant bar doesn't shift which opener is a
    # warmup — preserves the prior plates-based behavior.
    max_plates = max(s["plates"] for s in sess_sets)
    groups: dict = {}
    for s in sess_sets:
        dg = s.get("drop_group")
        if dg is not None:
            if dg not in groups:
                groups[dg] = {"parts": [], "first_plates": s["plates"]}
            part = f"{s['weight']} {unit} × {s['reps']} reps"
            if s.get("comment"):
                part += f" ({s['comment'].strip()})"
            groups[dg]["parts"].append(part)
    seen_groups: set = set()
    result = []
    set_num = 0
    for s in sess_sets:
        dg = s.get("drop_group")
        if dg is None:
            set_num += 1
            is_warmup = set_num == 1 and max_plates > 0 and s["plates"] < 0.6 * max_plates
            label = f"Set {set_num} (Warmup)" if is_warmup else f"Set {set_num}"
            set_display = f"{s['weight']} {unit} × {s['reps']} reps"
            if s.get("comment"):
                set_display += f" ({s['comment'].strip()})"
            result.append(f"{label}: {set_display}")
        elif dg not in seen_groups:
            seen_groups.add(dg)
            set_num += 1
            g = groups[dg]
            is_warmup = set_num == 1 and max_plates > 0 and g["first_plates"] < 0.6 * max_plates
            label = f"Set {set_num} (Warmup)" if is_warmup else f"Set {set_num}"
            indent = " " * len(f"{label}: ")
            joined = f"\n{indent}".join(g["parts"])
            result.append(f"{label}: {joined}")
    return result


# ── Shared display core (verbatim port of combined_server.py:2237-2349) ──────────
def _build_session_displays(raw_rows, ctx, exercise_name, is_cardio=False) -> list:
    """
    The SINGLE shared display core, called by both modes.

    raw_rows: rows carrying _id, date, typed_weight, reps, distance, duration_seconds,
    comment (same shape and aliases as the operational base_select; comment bound by FK
    at fetch time). Caller controls row ORDER (date DESC, _id ASC) — grouping preserves it.

    is_cardio: render distance/duration/pace lines instead of weight×reps (cardio rows
    store weight=reps=0).

    Returns [{date, unit, max_weight, total_sets, display_sets}] in the order the
    dates first appear in raw_rows (recency order for the standard ORDER BY).
    """
    # Group individual rows by date (caller's ORDER BY preserves recency order).
    # weight is BAR-INCLUSIVE (plates + date-ranged bar + offset) via the shared
    # conversion. plates (pre-bar) is kept alongside ONLY for the warmup ratio,
    # which must stay plates-based. unit is per-DATE, and every set in a session
    # shares its date, so a session has one well-defined unit.
    sessions_map: dict = {}
    for r in raw_rows:
        d = r["date"]
        w, unit, plates = _bar_inclusive_weight(ctx, exercise_name, d, r["typed_weight"])
        if d not in sessions_map:
            sessions_map[d] = []
        sessions_map[d].append({
            "set_db_id": r["_id"],                 # training_log._id
            "weight":    w,                        # bar-inclusive headline
            "plates":    plates,                   # plates-only, for warmup ratio
            "unit":      unit,
            "reps":      r["reps"],
            "distance":  r["distance"],            # cardio: km (may be 0/None)
            "duration_seconds": r["duration_seconds"],   # cardio: seconds (may be 0/None)
            "comment":   r["comment"],             # bound by id at fetch time (may be None)
        })

    sessions = [
        {
            "date": date,
            "sets": sets,
            "unit": sets[0]["unit"],               # all sets in a date share the unit
            "max_weight": max(s["weight"] for s in sets),
            "total_sets": len(sets),
        }
        for date, sets in sessions_map.items()
    ]

    for session in sessions:
        # Comment is already bound to its own set by id (Comment.owner_id =
        # training_log._id) from the fetch query — read it directly. The "Nth set"
        # drop-group label is parsed from the set's OWN comment.
        for s in session["sets"]:
            raw_comment = s.get("comment")
            s["drop_group"] = _parse_set_num(raw_comment)
            s["comment"] = _strip_set_label(raw_comment)

        session["display_sets"] = _build_display_sets(session["sets"], session["unit"], is_cardio)
        del session["sets"]

    return sessions


# ── Single-exercise mode (faithful port of _get_exercise_sessions_sync) ──────────
_BASE_SELECT = """
    SELECT tl._id, tl.date, tl.metric_weight AS typed_weight, tl.reps,
           tl.distance, tl.duration_seconds,
           c.comment AS comment
    FROM training_log tl
    LEFT JOIN Comment c ON c.owner_id = tl._id
    WHERE tl.exercise_id = :exercise_id
"""


def get_exercise_sessions(exercise_name: str, mode: str = "recent",
                          limit: int = 10, approximate_date: str = None,
                          date_from: str = None, date_to: str = None) -> dict:
    """
    Single-exercise session display. Faithful port of the operational MCP tool
    get_exercise_sessions (combined_server._get_exercise_sessions_sync): identical
    SQL, comment binding, bar-inclusive weights, drop-set grouping, warmup
    labeling, and verbatim display_sets strings.

    Returns a native dict (the analytical lane consumes Python structures); the
    MCP handler returns a JSON string of the SAME structure — display_sets content
    is identical.

    A "specific date" maps to mode="approximate" (±7 days) or mode="range" with
    date_from == date_to, matching the operational tool's surface.
    """
    conn = get_connection(DB_PATH)
    row = conn.execute(
        "SELECT _id, category_id FROM exercise WHERE name = ?", (exercise_name,)
    ).fetchone()
    if not row:
        return {"error": f"Exercise '{exercise_name}' not found."}
    exercise_id = row["_id"]
    is_cardio = CATEGORY_NAMES.get(row["category_id"]) == "Cardio"

    try:
        if mode == "recent":
            limit = min(int(limit), 20)
            raw_rows = conn.execute(
                _BASE_SELECT + " ORDER BY tl.date DESC, tl._id ASC",
                {"exercise_id": exercise_id},
            ).fetchall()
        elif mode == "approximate":
            raw_rows = conn.execute(
                _BASE_SELECT + """
                  AND tl.date BETWEEN date(:approximate_date, '-7 days')
                                  AND date(:approximate_date, '+7 days')
                ORDER BY tl.date DESC, tl._id ASC""",
                {"exercise_id": exercise_id, "approximate_date": approximate_date or ""},
            ).fetchall()
        elif mode == "range":
            raw_rows = conn.execute(
                _BASE_SELECT + " AND tl.date BETWEEN :date_from AND :date_to ORDER BY tl.date DESC, tl._id ASC",
                {"exercise_id": exercise_id, "date_from": date_from or "", "date_to": date_to or ""},
            ).fetchall()
        else:
            return {"error": f"Unknown mode '{mode}'. Use 'recent', 'approximate', or 'range'."}
    except Exception as exc:
        return {"error": str(exc)}

    try:
        ctx = load_user_context()
    except Exception:
        ctx = {}

    sessions = _build_session_displays(raw_rows, ctx, exercise_name, is_cardio)
    if mode == "recent":
        sessions = sessions[:limit]

    out = {
        "exercise": exercise_name,
        "mode": mode,
        "sessions": sessions,
        "count": len(sessions),
    }
    if exercise_name in BAR_EXERCISE_NOTES:
        # Weights are bar-inclusive, so the old "add the bar" note would be
        # double-counting. Keep the key (shape stable) but state it's included.
        out["bar_weight_note"] = (
            "Weights are BAR-INCLUSIVE (plates + bar already added) — do not add "
            "the bar again."
        )
    return out


# ── Category mode (NEW) ──────────────────────────────────────────────────────────
_CATEGORY_SELECT = """
    SELECT tl._id, tl.date, tl.metric_weight AS typed_weight, tl.reps,
           tl.distance, tl.duration_seconds,
           e.name AS exercise_name, c.comment AS comment
    FROM training_log tl
    JOIN exercise e ON tl.exercise_id = e._id
    LEFT JOIN Comment c ON c.owner_id = tl._id
    WHERE e.category_id = :cat_id AND tl.date = :target_date
    ORDER BY e.name, tl._id ASC
"""


def _empty_category(canon, category):
    return {"category": canon or category, "date": None, "exercises": [], "count": 0}


def _resolve_cat_id(category: str):
    """
    (canon, cat_id) for a muscle-group category. cat_id is None when the category
    is unknown OR is an excluded non-muscle category (Time/Place/Neck). canon is
    still returned when the name matched a muscle group, so callers can name it.
    """
    canon = match_muscle_group(category)
    if not canon:
        return None, None
    cat_id = {v: k for k, v in CATEGORY_NAMES.items()}.get(canon)
    if cat_id is None or cat_id in EXCLUDED_CATEGORY_IDS:
        return canon, None
    return canon, cat_id


def get_category_session(category: str, target: str = "recent") -> dict:
    """
    Category-level session display (NEW).

    Locked semantics: "last back session" = the single most recent date on which
    ANY exercise in that category was trained. Output one verbatim display block
    PER exercise trained on that date (identical to single-exercise mode for that
    date), grouped under per-exercise headers. ONE date only — never spans dates.

    target="recent" (default) resolves the most recent category date; otherwise
    target is a literal YYYY-MM-DD.

    Time/Place/Neck (ids 10/11/12) are not muscle-group categories and resolve to
    an empty result.
    """
    canon, cat_id = _resolve_cat_id(category)
    if cat_id is None:
        return _empty_category(canon, category)

    conn = get_connection(DB_PATH)
    _excl = ", ".join(str(c) for c in EXCLUDED_CATEGORY_IDS)

    if target is None or target == "recent":
        row = conn.execute(
            f"""SELECT MAX(tl.date) AS d
                FROM training_log tl
                JOIN exercise e ON tl.exercise_id = e._id
                WHERE e.category_id = ? AND e.category_id NOT IN ({_excl})""",
            (cat_id,),
        ).fetchone()
        target_date = row["d"] if row else None
    else:
        target_date = target

    if not target_date:
        return _empty_category(canon, category)

    rows = conn.execute(
        _CATEGORY_SELECT, {"cat_id": cat_id, "target_date": target_date}
    ).fetchall()
    if not rows:
        return {"category": canon, "date": None, "exercises": [], "count": 0}

    try:
        ctx = load_user_context()
    except Exception:
        ctx = {}

    # Group rows by exercise (ORDER BY e.name, tl._id preserves order), then run
    # the SAME shared core per exercise so each block is byte-identical to
    # single-exercise mode for this date.
    by_exercise: dict = {}
    for r in rows:
        by_exercise.setdefault(r["exercise_name"], []).append(r)

    cat_is_cardio = canon == "Cardio"
    exercises = []
    for ex_name, ex_rows in by_exercise.items():
        sess = _build_session_displays(ex_rows, ctx, ex_name, cat_is_cardio)[0]  # single date → one session
        block = {
            "exercise":    ex_name,
            "unit":        sess["unit"],
            "max_weight":  sess["max_weight"],
            "total_sets":  sess["total_sets"],
            "display_sets": sess["display_sets"],
        }
        if ex_name in BAR_EXERCISE_NOTES:
            block["bar_weight_note"] = (
                "Weights are BAR-INCLUSIVE (plates + bar already added) — do not add "
                "the bar again."
            )
        exercises.append(block)

    return {
        "category": canon,
        "date": target_date,
        "exercises": exercises,
        "count": len(exercises),
    }


# ── Approach (b): flat display_sets for the analytical package ────────────────────
# build_display_sets is the ONLY new entry point the package builder calls. It
# reuses get_exercise_sessions / get_category_session for the verbatim formatting
# (unchanged — their byte-equality tests stay valid), applies the ≤1 prior-session
# trigger, and flattens the result to a flat list[str] of self-contained header +
# per-set lines (the package's `display_sets` field).


def _exercise_prior_needed(sessions: list) -> bool:
    """Single-exercise ≤1 trigger (SETS-ONLY): the exercise-count is structurally
    1, so include the prior session iff the most-recent date had ≤1 set."""
    return bool(sessions) and sessions[0]["total_sets"] <= 1 and len(sessions) > 1


def _category_prior_needed(blocks: list) -> bool:
    """Category ≤1 trigger: include the prior category date iff the most-recent
    date was thin — ≤1 distinct exercise OR ≤1 set total across the date."""
    if not blocks:
        return False
    return len(blocks) <= 1 or sum(b["total_sets"] for b in blocks) <= 1


def _prior_category_date(category: str, before_date: str):
    """Next-most-recent date (< before_date) ANY exercise in this category was
    trained, or None when none exists."""
    _, cat_id = _resolve_cat_id(category)
    if cat_id is None:
        return None
    conn = get_connection(DB_PATH)
    _excl = ", ".join(str(c) for c in EXCLUDED_CATEGORY_IDS)
    row = conn.execute(
        f"""SELECT MAX(tl.date) AS d
            FROM training_log tl
            JOIN exercise e ON tl.exercise_id = e._id
            WHERE e.category_id = ? AND e.category_id NOT IN ({_excl})
              AND tl.date < ?""",
        (cat_id, before_date),
    ).fetchone()
    return row["d"] if row and row["d"] else None


def _flatten_exercise_sessions(sessions: list, exercise_name: str) -> list:
    """[{date, total_sets, display_sets}] → flat header + set-line strings."""
    out: list = []
    for s in sessions:
        out.append(f"{s['date']} — {exercise_name} ({s['total_sets']} sets):")
        out.extend(s["display_sets"])
    return out


def _flatten_category_blocks(date_str: str, category: str, blocks: list) -> list:
    """One date's per-exercise blocks → flat header + set-line strings."""
    out: list = [f"{date_str} — {category}:"]
    for b in blocks:
        out.append(f"{b['exercise']} ({b['total_sets']} sets):")
        out.extend(b["display_sets"])
        # bar_weight_note is an INTERNAL hint (kept as a dict key on the structured
        # return) — it must NOT enter the flat display_sets list, or the verbatim
        # DISPLAY SETS CHECK forces it into the user-facing answer.
    return out


def _build_category_display(target: str) -> tuple:
    """
    One category target's flat lines PLUS the exercise names contained in its
    displayed date(s) — the structural fact build_all_display_sets needs for the
    exercise-inside-category dedup. (flat_lines, contained_exercise_names).
    """
    r = get_category_session(target, "recent")
    blocks = r.get("exercises") or []
    if not blocks or not r.get("date"):
        return [], set()
    contained = {b["exercise"] for b in blocks}
    out = _flatten_category_blocks(r["date"], r["category"], blocks)
    if _category_prior_needed(blocks):
        prior = _prior_category_date(r["category"], r["date"])
        if prior:
            pr = get_category_session(r["category"], prior)
            pblocks = pr.get("exercises") or []
            if pblocks:
                out += _flatten_category_blocks(pr["date"], pr["category"], pblocks)
                contained |= {b["exercise"] for b in pblocks}
    return out, contained


def build_display_sets(kind: str, target: str) -> list:
    """
    The flat list[str] attached to the analytical package as `display_sets`.

      kind == "exercise"  → target is the resolved exercise name.
      kind == "category"  → target is the muscle-group name.

    "Last session" semantics: the single most-recent date in scope, PLUS the prior
    session when the ≤1 trigger fires (sets-only for an exercise; ≤1 exercise OR
    ≤1 set for a category). Returns [] when the scope has no logged sessions.
    """
    if kind == "exercise":
        r = get_exercise_sessions(target, mode="recent")
        sessions = r.get("sessions") or []
        if not sessions:
            return []
        included = sessions[:2] if _exercise_prior_needed(sessions) else sessions[:1]
        out = _flatten_exercise_sessions(included, r["exercise"])
        # bar_weight_note stays a dict key on r (structured return); never appended
        # to the flat list (the verbatim DISPLAY SETS CHECK would leak it to the user).
        return out

    if kind == "category":
        return _build_category_display(target)[0]

    return []


def build_all_display_sets(targets: list) -> list:
    """
    Flatten every display target into ONE display_sets list, with the
    exercise-inside-category dedup: an exercise target whose block already
    appears inside a built category block (the exercise was trained on one of
    the category's displayed dates) is emitted ONCE — the category copy is kept
    (it carries the date header and the same-day context, and containment
    implies the category block already shows that exercise's own most-recent
    session). An exercise NOT on any displayed category date keeps its
    standalone block — that is the only place its latest session shows.

    Output order preserves the historical flatten order: exercise blocks first,
    category blocks after. Exercise-only and category-only target lists are
    byte-identical to the per-target build_display_sets calls.
    """
    cat_lines: list = []
    contained: set = set()
    for kind, target in targets:
        if kind == "category":
            lines, names = _build_category_display(target)
            cat_lines.extend(lines)
            contained |= names

    flat: list = []
    for kind, target in targets:
        if kind == "exercise" and target not in contained:
            flat.extend(build_display_sets("exercise", target))
    flat.extend(cat_lines)
    return flat
