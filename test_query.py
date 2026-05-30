from src.data_agent import query

result = query(
    "SELECT e.name, tl.metric_weight, tl.reps "
    "FROM training_log tl "
    "JOIN exercise e ON tl.exercise_id = e._id "
    "WHERE e.name = 'Deadlift' "
    "ORDER BY tl._id DESC LIMIT 3"
)
for r in result["rows"]:
    print(r)
print()
print(result["warning"])
