from src.data_agent import collect

data = collect(query_period_days=365)
ranks = data["rankings"]

print("highest_volume (top 5):")
for r in ranks.get("highest_volume", [])[:5]:
    print(f"  {r['exercise']:40s}  {r['value']:.0f}")

print()
print("most_commented (top 5):")
for r in ranks.get("most_commented", [])[:5]:
    print(f"  {r['exercise']:40s}  {r['value']}")
