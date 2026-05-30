from src.data_agent import collect
import sys, time

# Test 2: Most-commented exercise with Phase 2
print("=== Test 2: Seated Machine Curl (Kg) + Phase2 ===")
start = time.time()
data = collect(exercise_names=["Seated Machine Curl (Kg)"], include_phase2=True)
ex = data["exercises"][0]
print(f"Time: {time.time()-start:.2f}s")
print(f"Size: {sys.getsizeof(str(data))/1024:.0f} KB")
print(f"Phase2 triggered: {ex['phase2_triggered']}")
print(f"Full comments: {len(ex.get('full_comments') or [])}")
print(f"Sessions: {len(ex.get('sessions') or [])}")
print()

# Test 3: Deadlift all-time Phase2 — unit switch + full comments
print("=== Test 3: Deadlift all-time + Phase2 ===")
start = time.time()
data = collect(exercise_names=["Deadlift"], query_period_days=None, include_phase2=True)
ex = data["exercises"][0]
p  = ex["progression"]
print(f"Time: {time.time()-start:.2f}s")
print(f"Size: {sys.getsizeof(str(data))/1024:.0f} KB")
print(f"Progression: {p['max_weight_start']} -> {p['max_weight_end']} {ex['unit']}")
print(f"Unit switch in period: {p['unit_switch_in_period']}")
print(f"Full comments: {len(ex.get('full_comments') or [])}")
print()

# Test 5: Single-day period — boundary edge case
print("=== Test 5: 1-day period ===")
start = time.time()
data = collect(query_period_days=1)
print(f"Exercises found: {data['total_exercises_analyzed']}")
print(f"Size: {sys.getsizeof(str(data))/1024:.0f} KB")
print(f"Time: {time.time()-start:.2f}s")
print("No crash: True")
