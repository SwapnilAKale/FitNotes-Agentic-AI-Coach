"""
The agent may add to your training data and manage what it added. It may never
modify or destroy what the FitNotes app recorded.

WHY THIS EXISTS. Two independent id sequences collide. The app holds ids 1-5; the
agent logs a new day into the LOCAL copy and takes id 6. Days later a fresh export
arrives in which the APP has assigned id 6 to a workout of its own — the app never
saw the agent's row, so its counter continued independently. Any WAL record for an
agent edit or delete still carries `set_id = 6`, so replay reaches for that id and
silently rewrites the user's real workout. `rowcount` does not help: the id exists,
it is simply the wrong row.

The fix is a rule, not a patch. Everything at or below the upload watermark came
from FitNotes and is untouchable; everything above it the agent created and may
manage. That makes the collision impossible rather than defended against — a WAL
record can no longer point at an app row, so replay only ever remaps its own
inserts.

Comments are the deliberate exception: a note about form is the agent's to write
even on a set the app logged.

EVERY TEST RUNS ON COPIES in tmp_path — never the real database.
"""

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import mcp_servers.combined_server as cs                       # noqa: E402
import src.settings as settings                                # noqa: E402
import src.wal as wal                                          # noqa: E402

_REAL_DB = _ROOT / "data" / "FitNotes_Backup.fitnotes"

WATERMARK_KEY = "app_data_watermark"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A copy of the real database plus isolated settings and WAL files."""
    if not _REAL_DB.exists():
        pytest.skip("real database not present")
    db = str(tmp_path / "copy.fitnotes")
    shutil.copy2(_REAL_DB, db)
    monkeypatch.setattr(cs, "DB_PATH", db)
    monkeypatch.setenv("SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(wal, "WAL_PATH", str(tmp_path / "wal.json"))
    cs._staged_writes.clear()
    return db


def _q(db, sql, args=()):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(sql, args)]
    con.close(); return out


def _newest(db):
    return _q(db, "SELECT * FROM training_log ORDER BY _id DESC LIMIT 1")[0]


def _name_of(db, exercise_id):
    return _q(db, "SELECT name FROM exercise WHERE _id=?", (exercise_id,))[0]["name"]


def _set_watermark(db):
    """Mark everything currently in the database as app-owned, the way /upload
    will once the new database has been written and before replay runs."""
    marks = {t: (_q(db, f"SELECT COALESCE(MAX(_id), 0) m FROM [{t}]")[0]["m"])
             for t in ("training_log", "Goal")}
    settings.set_setting(WATERMARK_KEY, marks)
    return marks


def _log_a_set(db, date="2026-08-08", weight=101, reps=7):
    """Agent-created row: staged, executed, id lands above the watermark."""
    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Flat Barbell Bench Press", "date": date,
        "sets": [{"weight": weight, "reps": reps, "unit": "lbs"}]}))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_workout_sync())
    assert out["success"] is True, out
    return _newest(db)


# ══════════════════════════════════════════════════════════════════════════════
# The rule — both directions
# ══════════════════════════════════════════════════════════════════════════════

def test_agent_may_delete_a_row_it_created(env):
    _set_watermark(env)
    row = _log_a_set(env)
    out = json.loads(cs._delete_workout_set_sync({
        "exercise_name": "Flat Barbell Bench Press", "date": row["date"],
        "weight": 101, "reps": 7, "unit": "lbs"}))
    assert out.get("staged") is True, out
    assert json.loads(cs._execute_staged_set_delete_sync())["success"] is True


def test_agent_may_edit_a_row_it_created(env):
    _set_watermark(env)
    row = _log_a_set(env)
    out = json.loads(cs._update_workout_set_sync({
        "exercise_name": "Flat Barbell Bench Press", "date": row["date"],
        "old_weight": 101, "old_reps": 7,
        "new_weight": 105, "new_reps": 7, "unit": "lbs"}))
    assert out.get("staged") is True, out
    assert json.loads(cs._execute_staged_set_update_sync())["success"] is True


def test_agent_may_not_delete_an_app_row(env):
    row = _newest(env)                       # exists before the watermark
    _set_watermark(env)
    out = json.loads(cs._delete_workout_set_sync({
        "exercise_name": _name_of(env, row["exercise_id"]), "date": row["date"],
        "weight": round(row["metric_weight"] * 2.2046, 1),
        "reps": row["reps"], "unit": "lbs"}))
    assert out.get("staged") is not True
    assert "error" in out
    assert "FitNotes" in out["error"]        # the refusal names why
    assert _q(env, "SELECT _id FROM training_log WHERE _id=?", (row["_id"],))


def test_agent_may_not_edit_an_app_row(env):
    row = _newest(env)
    _set_watermark(env)
    out = json.loads(cs._update_workout_set_sync({
        "exercise_name": _name_of(env, row["exercise_id"]), "date": row["date"],
        "old_weight": round(row["metric_weight"] * 2.2046, 1),
        "old_reps": row["reps"], "new_weight": 999, "new_reps": 1, "unit": "lbs"}))
    assert out.get("staged") is not True
    assert "FitNotes" in out.get("error", "")


def test_no_watermark_means_no_lock(env):
    """Before the first upload there is nothing to protect — a fresh install must
    not refuse everything."""
    settings.set_setting(WATERMARK_KEY, None)
    row = _newest(env)
    out = json.loads(cs._delete_workout_set_sync({
        "exercise_name": _name_of(env, row["exercise_id"]), "date": row["date"],
        "weight": round(row["metric_weight"] * 2.2046, 1),
        "reps": row["reps"], "unit": "lbs"}))
    assert out.get("staged") is True, out


# ══════════════════════════════════════════════════════════════════════════════
# The comment carve-out
# ══════════════════════════════════════════════════════════════════════════════

def test_a_comment_may_be_written_on_an_app_row(env):
    """The exception you asked for: talking about form and changing the note,
    even on a set FitNotes logged."""
    row = _newest(env)
    _set_watermark(env)
    out = json.loads(cs._set_set_comment_sync({
        "exercise_name": _name_of(env, row["exercise_id"]), "date": row["date"],
        "weight": round(row["metric_weight"] * 2.2046, 1), "reps": row["reps"],
        "unit": "lbs", "comment": "elbows flared on the last two"}))
    assert out.get("staged") is True, out
    assert json.loads(cs._execute_staged_set_comment_sync())["success"] is True
    got = _q(env, "SELECT comment FROM Comment WHERE owner_id=?", (row["_id"],))
    assert got and got[-1]["comment"] == "elbows flared on the last two"


def test_the_set_itself_is_still_untouchable_after_commenting(env):
    """A comment must not become a side door to editing the row it hangs on."""
    row = _newest(env)
    _set_watermark(env)
    cs._set_set_comment_sync({
        "exercise_name": _name_of(env, row["exercise_id"]), "date": row["date"],
        "weight": round(row["metric_weight"] * 2.2046, 1), "reps": row["reps"],
        "unit": "lbs", "comment": "noted"})
    cs._execute_staged_set_comment_sync()
    after = _q(env, "SELECT * FROM training_log WHERE _id=?", (row["_id"],))[0]
    assert after == row


# ══════════════════════════════════════════════════════════════════════════════
# Row 5 — the refusal must be machine-readable, not just readable
#
# Live, the model was refused three times and told the user why once. Twice it
# announced the refused edit as done ("I have successfully deleted your ... set
# of 100 lbs x 5 from July 22, 2026"), which the Coordinator's claim gate caught
# and replaced with a generic "nothing was written" — correct, but it left the
# user with no idea that FitNotes owns that row.
#
# So the refusal carries a tag. The Coordinator shows the sentence itself when
# the gate fires, and no longer depends on the model to pass it along.
# ══════════════════════════════════════════════════════════════════════════════

def test_the_refusal_is_tagged_for_the_coordinator(env):
    row = _newest(env)
    _set_watermark(env)
    out = json.loads(cs._delete_workout_set_sync({
        "exercise_name": _name_of(env, row["exercise_id"]), "date": row["date"],
        "weight": round(row["metric_weight"] * 2.2046, 1),
        "reps": row["reps"], "unit": "lbs"}))
    assert out["refused"] == "app_data_locked"
    assert "FitNotes" in out["error"]


def test_every_refusal_path_carries_the_tag():
    """All four locked tools go through _refuse_app_row, so the tag cannot be
    present on the set paths and missing on the goal ones."""
    for what in ("set", "goal"):
        assert json.loads(cs._refuse_app_row(what))["refused"] == "app_data_locked"


def test_the_refusal_reads_as_an_explanation_to_the_user(env):
    """It is shown VERBATIM to the user now, so it must not read as an internal
    error, and must say where the change has to be made instead."""
    msg = json.loads(cs._refuse_app_row("set"))["error"]
    assert "FitNotes" in msg
    assert "upload" in msg.lower()
    assert not msg.lower().startswith(("error", "failed", "exception"))


# ══════════════════════════════════════════════════════════════════════════════
# Row 7 — a verifier that can only say yes is not a verifier
#
# LIVE. After a confirmed goal deletion that never executed, the agent called
# verify_set_deleted — the wrong tool, for a GOAL — with weight=150, reps=1,
# date=2026-12-31. It asked "is there any training_log set matching this?",
# found none (of course), and returned "✅ Set confirmed deleted". The user was
# told their goal was gone while it sat in the database.
#
# Absence only proves deletion if the row was ever there, so verification is
# anchored to the id the delete actually targeted.
# ══════════════════════════════════════════════════════════════════════════════

def test_verify_set_deleted_confirms_a_real_deletion(env):
    _set_watermark(env)
    row = _log_a_set(env)
    assert json.loads(cs._delete_workout_set_sync({
        "exercise_name": "Flat Barbell Bench Press", "date": row["date"],
        "weight": 101, "reps": 7, "unit": "lbs"})).get("staged") is True
    assert json.loads(cs._execute_staged_set_delete_sync())["success"] is True

    out = json.loads(cs._verify_set_deleted_sync(
        "Flat Barbell Bench Press", row["date"], 101, 7, "lbs"))
    assert out["verified"] is True


def test_verify_set_deleted_will_not_confirm_what_it_never_deleted(env):
    """THE LIVE FAILURE. Nothing staged, nothing deleted — the honest answer is
    'cannot verify', never verified:true."""
    cs._staged_writes.clear()
    out = json.loads(cs._verify_set_deleted_sync(
        "Flat Barbell Bench Press", "2026-12-31", 150, 1, "lbs"))
    assert out["verified"] is False
    assert out.get("unverifiable") is True
    assert "does NOT mean anything was deleted" in out["message"]


def test_verify_set_deleted_reports_a_delete_that_did_not_take(env):
    """Other direction — a row still present must fail verification."""
    _set_watermark(env)
    row = _log_a_set(env)
    cs._staged_writes.clear()
    cs._staged_writes["last_deleted_set"] = {"set_id": row["_id"]}
    out = json.loads(cs._verify_set_deleted_sync(
        "Flat Barbell Bench Press", row["date"], 101, 7, "lbs"))
    assert out["verified"] is False
    assert "still exists" in out["message"]


def test_bookkeeping_keys_are_not_pending_writes(env):
    """last_deleted_set and the stage-2 verdict ride in _staged_writes so the
    turn-start discard wipes them for free — but counting them as pending would
    resurrect the 'discarded: true when nothing was staged' dishonesty."""
    cs._staged_writes.clear()
    cs._staged_writes["last_deleted_set"] = {"set_id": 1}
    cs._staged_writes["workout_verify"] = {"verdict": "PASS", "reason": ""}
    out = json.loads(cs._discard_staged_writes_sync())
    assert out["had_pending"] is False
    assert out["discarded"] is False


def test_a_real_staged_write_still_counts_as_pending(env):
    cs._staged_writes.clear()
    cs._staged_writes["delete_set"] = {"set_id": 1}
    out = json.loads(cs._discard_staged_writes_sync())
    assert out["had_pending"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Row A - bodyweight was the only logging operation without CRUD
#
# Four symptoms, one cause: it never got the staged/execute shape. No update and
# no delete, so "delete both bodyweight logs" was refused for want of the tools
# and the rows had to be removed with SQL. Insert-only, so logging body fat
# after a weight produced a SECOND weigh-in for the same day (live: 180 lbs/0.0
# and 180 lbs/21.95, both 2026-09-11). A raw JSON confirm panel, because the
# renderers read the staged slot and a direct write had none. And body_fat 0.0
# stored as a number when it means "not measured".
# ══════════════════════════════════════════════════════════════════════════════

def _bw(db):
    return _q(db, "SELECT * FROM BodyWeight ORDER BY _id")


def _log_bw(**kw):
    staged = json.loads(cs._log_bodyweight_sync(kw))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_bodyweight_sync())
    assert out["success"] is True, out
    return out


def test_a_first_weigh_in_inserts(env):
    _set_watermark(env)
    _log_bw(body_weight=180, unit="lbs")
    rows = _bw(env)
    assert len(rows) == 1
    assert round(rows[0]["body_weight_metric"] * 2.2046) == 180


def test_logging_body_fat_after_a_weight_UPDATES_it(env):
    """THE live duplicate. Two weigh-ins for one day, one carrying body fat and
    one carrying the 0 sentinel, both 180 lbs."""
    _set_watermark(env)
    _log_bw(body_weight=180, unit="lbs")
    _log_bw(body_fat_percent=21.95, unit="lbs")       # no weight - the live phrasing
    rows = _bw(env)
    assert len(rows) == 1, f"logging body fat added a second weigh-in: {rows}"
    assert rows[0]["body_fat"] == 21.95
    assert round(rows[0]["body_weight_metric"] * 2.2046) == 180


def test_a_second_weight_on_the_same_day_updates_rather_than_duplicating(env):
    _set_watermark(env)
    _log_bw(body_weight=180, unit="lbs")
    _log_bw(body_weight=182, unit="lbs")
    rows = _bw(env)
    assert len(rows) == 1
    assert round(rows[0]["body_weight_metric"] * 2.2046) == 182


def test_body_fat_alone_with_no_weight_for_the_day_asks_rather_than_guessing(env):
    """The other direction - there is no weight to attach it to, and inventing
    one would be worse than asking."""
    _set_watermark(env)
    out = json.loads(cs._log_bodyweight_sync({"body_fat_percent": 21.95, "unit": "lbs"}))
    assert out.get("needs_clarification") is True
    assert out.get("staged") is not True
    assert _bw(env) == []


def test_bodyweight_can_be_deleted(env):
    """The operation that was impossible - the agent refused because the tool
    did not exist, and the rows had to be removed with SQL."""
    _set_watermark(env)
    _log_bw(body_weight=180, unit="lbs")
    staged = json.loads(cs._delete_bodyweight_sync({}))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_bodyweight_delete_sync())
    assert out["success"] is True, out
    assert _bw(env) == []


def test_deleting_a_day_with_no_weigh_in_is_refused(env):
    _set_watermark(env)
    out = json.loads(cs._delete_bodyweight_sync({"date": "2019-01-01"}))
    assert "error" in out
    assert out.get("staged") is not True


# -- the panel, rendered from the slot -----------------------------------------

def test_the_bodyweight_panel_reads_as_a_sentence(env):
    _set_watermark(env)
    cs._log_bodyweight_sync({"body_weight": 180, "unit": "lbs", "body_fat_percent": 21.95})
    out = json.loads(cs._format_staged_write_for_confirmation_sync())
    preview = out["preview"]
    assert out["staged_key"] == "bodyweight"
    assert "180" in preview and "21.95" in preview
    assert "{" not in preview and '":' not in preview


def test_the_panel_hides_the_not_measured_sentinel(env):
    """0 is 'not measured', so showing 'Body fat: 0%' on the panel would be
    asking the user to approve a reading that does not exist."""
    _set_watermark(env)
    cs._log_bodyweight_sync({"body_weight": 180, "unit": "lbs"})
    preview = json.loads(cs._format_staged_write_for_confirmation_sync())["preview"]
    assert "Body fat" not in preview


def test_the_delete_panel_names_the_entry(env):
    _set_watermark(env)
    _log_bw(body_weight=180, unit="lbs")
    cs._delete_bodyweight_sync({})
    out = json.loads(cs._format_staged_write_for_confirmation_sync())
    assert out["staged_key"] == "delete_bodyweight"
    assert "180" in out["preview"]


# -- the sentinel, translated at the READ boundary -----------------------------

def test_a_zero_body_fat_reads_back_as_not_measured(env):
    """body_fat 0 is a NOT NULL placeholder, never a reading. It must not leave
    the database as a number, or an average folds a non-measurement into a
    trend."""
    from src.data_agent.fetch import _fetch_all_bodyweight
    con = sqlite3.connect(env)
    con.row_factory = sqlite3.Row
    con.execute("INSERT INTO BodyWeight (date, body_weight_metric, body_fat) "
                "VALUES ('2026-09-11', 81.6, 0)")
    con.execute("INSERT INTO BodyWeight (date, body_weight_metric, body_fat) "
                "VALUES ('2026-09-12', 81.6, 21.95)")
    con.commit()
    rows = _fetch_all_bodyweight(con)
    con.close()
    by_date = {r["date"]: r["body_fat"] for r in rows}
    assert by_date["2026-09-11"] is None, "the sentinel reached analysis as a reading"
    assert by_date["2026-09-12"] == 21.95, "a real measurement was lost"
