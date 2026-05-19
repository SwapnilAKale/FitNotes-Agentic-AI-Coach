import json
import os

_SCHEMA = """
## FitNotes Database Schema

### Tables

#### training_log
One row per set performed. This is the primary table.
| Column              | Type    | Notes                                      |
|---------------------|---------|--------------------------------------------|
| _id                 | INTEGER | Primary key                                |
| exercise_id         | INTEGER | FK → exercise._id                          |
| date                | TEXT    | ISO format: YYYY-MM-DD                     |
| metric_weight       | REAL    | Weight in **kilograms**                    |
| reps                | INTEGER | Repetitions performed                      |
| distance            | INTEGER | Distance in meters (cardio exercises)      |
| duration_seconds    | INTEGER | Duration in seconds (cardio / timed sets)  |
| is_personal_record  | INTEGER | 1 if this set was a PR, 0 otherwise        |

#### exercise
| Column      | Type    | Notes                                   |
|-------------|---------|-----------------------------------------|
| _id         | INTEGER | Primary key                             |
| name        | TEXT    | Exercise name (e.g., 'Lat Pulldown')    |
| category_id | INTEGER | FK → Category._id                       |

#### Category
| Column | Type    | Notes                                                                      |
|--------|---------|----------------------------------------------------------------------------|
| _id    | INTEGER | Primary key                                                                |
| name   | TEXT    | One of: Shoulders, Triceps, Biceps, Chest, Back, Legs, Abs, Cardio,       |
|        |         | Forearms, Time, Place, Neck                                                |

#### BodyWeight  *(may be empty)*
| Column              | Type    | Notes                        |
|---------------------|---------|------------------------------|
| _id                 | INTEGER | Primary key                  |
| date                | TEXT    | YYYY-MM-DD                   |
| body_weight_metric  | REAL    | Body weight in kilograms     |
| body_fat            | REAL    | Body fat percentage          |

#### Goal  *(may be empty)*
| Column        | Type    | Notes                      |
|---------------|---------|----------------------------|
| _id           | INTEGER | Primary key                |
| exercise_id   | INTEGER | FK → exercise._id          |
| metric_weight | REAL    | Target weight in kg        |
| reps          | INTEGER | Target reps                |
| title         | TEXT    | Goal description           |
| target_date   | TEXT    | YYYY-MM-DD                 |
| start_date    | TEXT    | YYYY-MM-DD                 |

### Key Relationships
- `training_log.exercise_id` → `exercise._id`
- `exercise.category_id` → `Category._id`
- `Goal.exercise_id` → `exercise._id`

### Important Rules
1. All weights (`metric_weight`) are in **kilograms**.
2. The database stores metric_weight in kilograms (kg). FitNotes divides every typed number by 2.2046 before storing. To recover the original typed value: metric_weight * 2.2046. Apply unit conversion INSIDE the SQL using the patterns in the User Data Conventions section below. Name weight columns weight_lbs or weight_kg so the answer layer knows the unit.
2. Dates are ISO strings (`YYYY-MM-DD`). Use `date('now', '-N days')` for relative ranges.
3. A **set** = one row in `training_log`. A **workout** = all rows sharing the same `date`.
4. Only generate `SELECT` queries. Never use INSERT, UPDATE, DELETE, DROP, or CREATE.
5. Always use `LIMIT` unless you are aggregating into a single summary row.
6. Join `training_log` → `exercise` → `Category` when filtering by muscle group name.
7. Exercise names are case-sensitive and match exactly (e.g., `'Lat Pulldown'`, `'Flat Dumbbell Bench Press'`).

IMPORTANT: All primary key and foreign key columns in this database use an underscore prefix.
Always write: exercise._id, Category._id, training_log._id, exercise.category_id.
Never write e.id or c.id — those columns do not exist and will silently return no results.

### Example Queries

-- 1. Personal record (heaviest set) for Lat Pulldown:
SELECT tl.metric_weight AS pr_kg, tl.reps, tl.date
FROM training_log tl
JOIN exercise e ON tl.exercise_id = e._id
WHERE e.name = 'Lat Pulldown' AND tl.is_personal_record = 1
ORDER BY tl.metric_weight DESC
LIMIT 1;

-- 2. Total sets per muscle group in the last 30 days:
SELECT c.name AS muscle_group, COUNT(*) AS total_sets
FROM training_log tl
JOIN exercise e ON tl.exercise_id = e._id
JOIN Category c ON e.category_id = c._id
WHERE tl.date >= date('now', '-30 days')
GROUP BY c.name
ORDER BY total_sets DESC;

-- 3. Weight progression for Flat Dumbbell Bench Press (most recent 20 sessions):
SELECT tl.date, MAX(tl.metric_weight) AS max_weight_kg, SUM(tl.reps) AS total_reps
FROM training_log tl
JOIN exercise e ON tl.exercise_id = e._id
WHERE e.name = 'Flat Dumbbell Bench Press'
GROUP BY tl.date
ORDER BY tl.date DESC
LIMIT 20;
"""


def load_user_context(path: str = "data/user_context.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: user_context.json not found at {path}. User context will not be injected.")
        return {}


def build_user_context_prompt(ctx: dict) -> str:
    if not ctx:
        return ""

    return """\
## User Data Conventions

UNIT RECOVERY -- HOW FITNOTES STORES DATA:
FitNotes treats every typed number as lbs and stores it as kg (divides by 2.2046).
To recover the original number the user typed: metric_weight * 2.2046 = typed value.

For lbs-native exercises (everything except the 4 below):
  metric_weight * 2.2046 = plate weight in lbs
  Bar weight for these exercises must also be expressed in lbs: bar_kg * 2.2046
  SQL pattern (with bar): (metric_weight + bar_kg) * 2.2046 AS weight_lbs
  SQL pattern (no bar): metric_weight * 2.2046 AS weight_lbs

For kg-native exercises (Seated Machine Curl (Kg), Machine Wrist Extension, Hand Gripper,
and Deadlift from 2025-12-26 onwards):
  metric_weight * 2.2046 = plate weight in kg
  Bar weight for these exercises stays in kg
  SQL pattern (with bar): (metric_weight * 2.2046) + bar_kg AS weight_kg
  SQL pattern (no bar): metric_weight * 2.2046 AS weight_kg

Deadlift special case:
  Before 2025-12-26: lbs-native. Use: (metric_weight * 2.2046) + (20 * 2.2046) AS weight_lbs
  From 2025-12-26 onwards: kg-native. Use: (metric_weight * 2.2046) + 20 AS weight_kg

Machine Wrist Extension special case:
  If metric_weight = 0, treat plate weight as 5 kg.
  SQL: CASE WHEN metric_weight = 0 THEN 5 ELSE metric_weight * 2.2046 END AS weight_kg

Hand Gripper special case:
  Usually kg-native. But if the comment for a specific set contains the word 'Pounds',
  that set was typed in lbs. For PR queries, exclude sets with 'Pounds' in comment
  or handle separately.

BAR WEIGHT ADDITION RULES:
Always use (metric_weight + bar_kg) * 2.2046 for lbs exercises -- never add bar after conversion.
Always use (metric_weight * 2.2046) + bar_kg for kg exercises -- never convert bar weight.

Bar weights by exercise and date (these are in kg):
  Barbell Curl: before 2024-09-24 add 10, from 2024-09-24 to 2025-10-30 add 12.5, from 2025-10-31 add 15
  Barbell Upright Row: before 2025-01-21 add 10, from 2025-01-21 to 2025-08-04 add 12.5, from 2025-08-05 add 15
  Behind The Back Wrist Curls: before 2024-11-22 add 10, from 2024-11-22 to 2025-07-18 add 12.5, from 2025-07-19 add 15
  EZ-Bar Curl: always add 10
  Reverse Zig Zag Barbell Curls: always add 10
  Barbell Row: always add 20
  Deadlift (kg period): always add 20
  All Smith Machine exercises: always add 20 (bar), weights are lbs-native
  Smith Machine counterbalance: 'one support' in comment = subtract 10 from bar. 'two supports' = subtract 20 from bar.

SET STRUCTURE:
  Drop sets: sets with same number in comment (e.g. '2nd set', '3rd set') on same exercise/date = back-to-back, no rest
  Warmups to exclude: Flat Dumbbell Bench Press sets commented 'Done first as a warmup',
  dumbbell skull crusher sets at typed 7.5 (stored 3.4kg) commented 'Warmup for elbows',
  Cable Triceps Extension sets commented 'Warmup'\
"""


def build_schema_prompt() -> str:
    ctx = load_user_context()
    schema = _SCHEMA.strip()
    user_ctx_block = build_user_context_prompt(ctx)
    if user_ctx_block:
        return schema + "\n\n" + user_ctx_block
    return schema
