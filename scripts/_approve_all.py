import json
with open("evals/candidates.json") as f:
    c = json.load(f)
for v in c.values():
    v["approved"] = True
with open("evals/candidates.json", "w") as f:
    json.dump(c, f, indent=2)
print("Approved " + str(len(c)) + " candidates.")
