import json
import random

with open("spider_codex_conversion_k_20_n_1969.jsonl") as f:
    data = [json.loads(item) for item in list(f)]

print("Num. Examples:", len(data))
successes = []
fails = []
for item in data:
    truthy_results = [x[1] for x in item["program_result_list"]]
    if True in truthy_results:
        successes.append(item)
    elif False in truthy_results:
        fails.append(item)
    else:
        raise Exception("Result neither true or false")
print("Sucesses:", len(successes))
print("Fails:", len(fails))

analyze_success = random.sample(successes, 15)
analyze_fail = random.sample(fails, 15)

with open("fifteen_successes.jsonl", "w") as out:
    out.write("\n".join([json.dumps(json_og) for json_og in analyze_success]))
with open("fifteen_fails.jsonl", "w") as out:
    out.write("\n".join([json.dumps(json_og) for json_og in analyze_fail]))


