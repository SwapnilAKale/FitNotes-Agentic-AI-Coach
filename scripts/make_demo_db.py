#!/usr/bin/env python3
"""
Generate a SYNTHETIC FitNotes backup for the public demo.

    python scripts/make_demo_db.py                       # → data/FitNotes_Backup.fitnotes
    python scripts/make_demo_db.py --out /tmp/demo.fitnotes --weeks 12 --seed 7
    python scripts/make_demo_db.py --end-date 2026-10-30 --force

Everything in the file is invented by this script: a made-up lifter running a
push / pull / legs split for ~12 weeks, with double progression (add reps up to
the top of the range, then add weight), a deload week, the odd missed session,
a few set comments, weekly body-weight entries and two goals. No real person's
data is read or reproduced. Same --seed and --end-date → byte-identical rows.

SCHEMA — the subset of the FitNotes Android backup that this codebase reads
(src/schema_prompt.py, src/db.py, src/data_agent/fetch.py, src/wal.py,
mcp_servers/combined_server.py, and the tests' fixtures), plus the app's own
NOT NULL DEFAULT columns so the write path's INSERTs work unchanged:

  Category(_id, name, colour, sort_order)          ids 1-8 = CATEGORY_NAMES
  exercise(_id, name, category_id, exercise_type_id, ...)
  training_log(_id, exercise_id, date, metric_weight, reps, unit,
               is_personal_record, is_complete, distance, duration_seconds, ...)
  Comment(_id, date, owner_type_id, owner_id, comment)   owner_type_id 1 = a set
  Goal(_id, type_id, exercise_id, metric_weight, reps, unit, title,
       target_date, sort_order, distance, duration_seconds, start_date)
  BodyWeight(_id, date, body_weight_metric, body_fat, comments)

CONVENTIONS the code assumes (src/schema_prompt.py "User Data Conventions"):
  * Every typed weight is in POUNDS and stored as metric_weight = lbs / 2.2046.
  * Exercise names come from ontology/aliases.csv so the muscle graph maps them,
    and avoid the names that carry hard-coded special rules there (Deadlift,
    Barbell Row, Smith Machine *, the kg-native ones, ...). Machine and dumbbell
    weights are what the stack/dumbbell says; barbell weights include the bar.
  * Strength rows: unit=0, distance=0, duration_seconds=0.
    Distance cardio rows: metric_weight=0, reps=0, unit=3, distance (km), duration.
  * is_personal_record = 1 when a set's weight beats every earlier set of that
    exercise — the same rule the write path applies when it logs a set.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import sqlite3
import sys
from pathlib import Path

LBS_PER_KG = 2.2046

CATEGORIES = [  # (_id, name, colour) — ids match src/data_agent/process.py
    (1, "Shoulders", -11751600), (2, "Triceps", -14273992), (3, "Biceps", -5317),
    (4, "Chest", -1499549), (5, "Back", -12627531), (6, "Legs", -16738680),
    (7, "Abs", -6543440), (8, "Cardio", -8825528),
]

# name, category_id, start lbs, increment lbs, rep range, sets, rest seconds
EXERCISES = {
    "Flat Barbell Bench Press":     (4, 135, 5,   (6, 8),   4, 180),
    "Incline Dumbbell Bench Press": (4, 45,  5,   (8, 12),  3, 120),
    "Overhead Press":               (1, 85,  5,   (6, 8),   3, 150),
    "Lateral Dumbbell Raise":       (1, 15,  2.5, (12, 15), 3, 60),
    "Seated Machine Fly":           (4, 90,  10,  (10, 12), 3, 90),
    "Cable Triceps Extension":      (2, 40,  5,   (10, 12), 3, 75),

    "Lat Pulldown":                 (5, 120, 10,  (8, 10),  4, 120),
    "Wide Grip Cable Row":          (5, 110, 10,  (8, 12),  3, 120),
    "Single Hand Dumbbell Row":     (5, 55,  5,   (8, 12),  3, 90),
    "Cable Face Pull":              (1, 40,  5,   (12, 15), 3, 60),
    "Dumbbell Curl":                (3, 25,  5,   (8, 12),  3, 75),
    "Dumbbell Hammer Curl":         (3, 30,  5,   (8, 12),  3, 75),

    "Leg Press":                    (6, 270, 20,  (8, 12),  4, 150),
    "Lying Hip Thrust":             (6, 135, 10,  (8, 12),  3, 120),
    "Hamstring Curls Machine":      (6, 70,  10,  (10, 12), 3, 90),
    "Leg Extension Machine":        (6, 80,  10,  (10, 12), 3, 90),
    "Calf Raises":                  (6, 100, 10,  (12, 15), 3, 60),
    "Cable Crunch":                 (7, 60,  10,  (12, 15), 3, 60),
}
CARDIO = ("Treadmill", 8)

DAYS = {
    "push": ["Flat Barbell Bench Press", "Incline Dumbbell Bench Press", "Overhead Press",
             "Lateral Dumbbell Raise", "Seated Machine Fly", "Cable Triceps Extension"],
    "pull": ["Lat Pulldown", "Wide Grip Cable Row", "Single Hand Dumbbell Row",
             "Cable Face Pull", "Dumbbell Curl", "Dumbbell Hammer Curl"],
    "legs": ["Leg Press", "Lying Hip Thrust", "Hamstring Curls Machine",
             "Leg Extension Machine", "Calf Raises", "Cable Crunch"],
}
# Five training days a week (Mon, Tue, Wed, Fri, Sat) cycling push → pull → legs
# continuously, so each day type comes round ~1.7 times a week (a rolling PPL).
WEEK_SLOTS = [0, 1, 2, 4, 5]          # weekday() of training days
ROTATION = ["push", "pull", "legs"]

SET_COMMENTS = [
    "Felt strong today", "Paused first rep", "Grip gave out", "Slow eccentric",
    "Last rep was a grind", "Slight elbow ache, kept it light", "Easy, go up next time",
]

SCHEMA_SQL = """
CREATE TABLE android_metadata (locale TEXT);
CREATE TABLE Category (
    _id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
    colour INTEGER NOT NULL DEFAULT 0, sort_order INTEGER NOT NULL DEFAULT 0);
CREATE TABLE exercise (
    _id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
    category_id INTEGER NOT NULL, exercise_type_id INTEGER NOT NULL DEFAULT 0,
    notes TEXT, weight_increment INTEGER, default_graph_id INTEGER,
    default_rest_time INTEGER, weight_unit_id INTEGER NOT NULL DEFAULT 0,
    is_favourite INTEGER NOT NULL DEFAULT 0);
CREATE TABLE training_log (
    _id INTEGER PRIMARY KEY AUTOINCREMENT, exercise_id INTEGER NOT NULL,
    date DATE NOT NULL, metric_weight REAL NOT NULL DEFAULT 0,
    reps INTEGER NOT NULL DEFAULT 0, unit INTEGER NOT NULL DEFAULT 0,
    routine_section_exercise_set_id INTEGER NOT NULL DEFAULT 0,
    timer_auto_start INTEGER NOT NULL DEFAULT 0,
    is_personal_record INTEGER NOT NULL DEFAULT 0,
    is_personal_record_first INTEGER NOT NULL DEFAULT 0,
    is_complete INTEGER NOT NULL DEFAULT 0,
    is_pending_update INTEGER NOT NULL DEFAULT 0,
    distance REAL NOT NULL DEFAULT 0, duration_seconds INTEGER NOT NULL DEFAULT 0);
CREATE INDEX training_log_exercise_date ON training_log (exercise_id, date);
CREATE TABLE Comment (
    _id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE NOT NULL,
    owner_type_id INTEGER NOT NULL, owner_id INTEGER NOT NULL, comment TEXT NOT NULL);
CREATE TABLE Goal (
    _id INTEGER PRIMARY KEY AUTOINCREMENT, type_id INTEGER NOT NULL DEFAULT 1,
    exercise_id INTEGER NOT NULL, metric_weight REAL NOT NULL DEFAULT 0,
    reps INTEGER NOT NULL DEFAULT 0, unit INTEGER NOT NULL DEFAULT 0, title TEXT,
    target_date TEXT, sort_order INTEGER NOT NULL DEFAULT 0,
    distance REAL NOT NULL DEFAULT 0, duration_seconds INTEGER NOT NULL DEFAULT 0,
    start_date TEXT);
CREATE TABLE BodyWeight (
    _id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE NOT NULL,
    body_weight_metric REAL NOT NULL, body_fat REAL NOT NULL DEFAULT 0, comments TEXT);
"""


def _kg(lbs: float) -> float:
    return round(lbs / LBS_PER_KG, 6)


def _round_to(x: float, step: float) -> float:
    return round(round(x / step) * step, 2)


class _Lift:
    """Double progression for one exercise: reps climb to the top of the range,
    then weight goes up one increment and reps drop back to the bottom."""

    def __init__(self, name: str, rng: random.Random):
        cat, start, inc, (lo, hi), sets, _rest = EXERCISES[name]
        self.name, self.inc, self.lo, self.hi, self.sets = name, inc, lo, hi, sets
        self.weight = float(start)
        self.target = lo
        self.rng = rng

    def session(self, deload: bool, bad_day: bool) -> list[tuple[float, int]]:
        rng = self.rng
        if deload:
            w = max(self.inc, _round_to(self.weight * 0.8, self.inc))
            return [(w, self.lo)] * max(2, self.sets - 1)
        out = []
        for i in range(self.sets):
            reps = self.target - (1 if i >= self.sets - 1 and rng.random() < 0.5 else 0)
            reps -= 2 if bad_day else 0
            reps += 1 if rng.random() < 0.12 else 0
            out.append((self.weight, max(1, min(self.hi + 1, reps))))
        if not bad_day:
            if all(r >= self.hi for _, r in out):
                self.weight += self.inc
                self.target = self.lo
            elif rng.random() < 0.7:
                self.target = min(self.hi, self.target + 1)
        return out


def generate(path: str, weeks: int = 12, end_date: dt.date | None = None,
             seed: int = 42) -> dict:
    rng = random.Random(seed)
    end = end_date or (dt.date.today() - dt.timedelta(days=1))
    # Start on the Monday `weeks` weeks before the end date.
    start = end - dt.timedelta(days=end.weekday()) - dt.timedelta(weeks=weeks - 1)
    deload_week = weeks // 2 + 1 if weeks >= 6 else None

    if os.path.exists(path):
        os.unlink(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("INSERT INTO android_metadata VALUES ('en_US')")
    conn.executemany("INSERT INTO Category (_id, name, colour, sort_order) VALUES (?,?,?,?)",
                     [(i, n, c, i) for i, n, c in CATEGORIES])

    ex_ids: dict[str, int] = {}
    for name, (cat, _s, _inc, _r, _n, rest) in EXERCISES.items():
        cur = conn.execute(
            "INSERT INTO exercise (name, category_id, exercise_type_id, default_rest_time)"
            " VALUES (?,?,0,?)", (name, cat, rest))
        ex_ids[name] = cur.lastrowid
    cur = conn.execute("INSERT INTO exercise (name, category_id, exercise_type_id)"
                       " VALUES (?,?,3)", CARDIO)
    ex_ids[CARDIO[0]] = cur.lastrowid

    lifts = {n: _Lift(n, rng) for n in EXERCISES}
    best: dict[str, float] = {}
    rot = 0
    sessions = sets_written = 0
    day = start
    while day <= end:
        week = (day - start).days // 7 + 1
        if day.weekday() in WEEK_SLOTS and rng.random() > 0.07:      # ~7% missed
            kind = ROTATION[rot % 3]
            rot += 1
            deload = week == deload_week
            bad_day = (not deload) and rng.random() < 0.06
            ds = day.isoformat()
            for name in DAYS[kind]:
                if name in ("Cable Crunch", "Calf Raises") and rng.random() < 0.25:
                    continue                                           # skipped accessory
                for w_lbs, reps in lifts[name].session(deload, bad_day):
                    mw = _kg(w_lbs)
                    is_pr = 1 if mw > best.get(name, 0.0) + 1e-9 else 0
                    if is_pr:
                        best[name] = mw
                    cur = conn.execute(
                        "INSERT INTO training_log (exercise_id, date, metric_weight, reps,"
                        " unit, is_personal_record, is_complete, distance, duration_seconds)"
                        " VALUES (?,?,?,?,0,?,1,0,0)", (ex_ids[name], ds, mw, reps, is_pr))
                    sets_written += 1
                    if rng.random() < 0.025:
                        conn.execute(
                            "INSERT INTO Comment (date, owner_type_id, owner_id, comment)"
                            " VALUES (?,1,?,?)", (ds, cur.lastrowid, rng.choice(SET_COMMENTS)))
            if kind == "legs" and rng.random() < 0.6:                  # cool-down walk/jog
                km = round(rng.uniform(1.6, 3.2), 2)
                secs = int(km * rng.uniform(420, 560))
                conn.execute(
                    "INSERT INTO training_log (exercise_id, date, metric_weight, reps, unit,"
                    " is_personal_record, is_complete, distance, duration_seconds)"
                    " VALUES (?,?,0,0,3,0,1,?,?)", (ex_ids[CARDIO[0]], ds, km, secs))
                sets_written += 1
            sessions += 1
        if day.weekday() == 0:                                         # Monday weigh-in
            bw = 78.0 + 0.09 * (day - start).days / 7 + rng.uniform(-0.4, 0.4)  # slow gain
            conn.execute("INSERT INTO BodyWeight (date, body_weight_metric, body_fat)"
                         " VALUES (?,?,0)", (day.isoformat(), round(bw, 1)))
        day += dt.timedelta(days=1)

    for name, lbs, reps, title in (("Flat Barbell Bench Press", 185, 5, "Bench 185 x 5"),
                                   ("Leg Press", 450, 8, "Leg press 450 x 8")):
        conn.execute(
            "INSERT INTO Goal (type_id, exercise_id, metric_weight, reps, unit, title,"
            " target_date, sort_order, distance, duration_seconds, start_date)"
            " VALUES (1,?,?,?,0,?,?,0,0,0,?)",
            (ex_ids[name], _kg(lbs), reps, title,
             (end + dt.timedelta(weeks=10)).isoformat(), start.isoformat()))
    conn.commit()
    conn.close()
    return {"path": path, "start": start.isoformat(), "end": end.isoformat(),
            "sessions": sessions, "rows": sets_written, "deload_week": deload_week}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.environ.get(
        "FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes"))
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--end-date", type=dt.date.fromisoformat, default=None,
                    help="last possible training day (default: yesterday)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} exists — pass --force to overwrite it.", file=sys.stderr)
        return 1
    info = generate(a.out, weeks=a.weeks, end_date=a.end_date, seed=a.seed)
    print(f"Demo DB written: {info['path']}  ({info['start']} → {info['end']}, "
          f"{info['sessions']} sessions, {info['rows']} log rows, "
          f"deload week {info['deload_week']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
