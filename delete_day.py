import sqlite3

DB = "data/FitNotes_Backup.fitnotes"
DATE = "2026-06-29"

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("DELETE FROM Comment WHERE date = ?", (DATE,))
comments_deleted = cur.rowcount

cur.execute("DELETE FROM training_log WHERE date = ?", (DATE,))
sets_deleted = cur.rowcount

conn.commit()
conn.close()

print(f"Deleted {comments_deleted} Comment rows and {sets_deleted} training_log rows for {DATE}.")