import json

if __name__ == "__main__":
    with open("spider_codex_conversion_k_1_n_6997.jsonl") as f:
        sample_1 = list(f)
    sample_1 = [json.loads(json_str) for json_str in sample_1]
    
    print("Num. Examples:", len(sample_1))
    successes = []
    fails = []
    for item in sample_1:
        truthy_results = [x[1] for x in item["program_result_list"]]
        if True in truthy_results:
            successes.append(item)
        elif False in truthy_results:
            fails.append(item)
        else:
            raise Exception("Result neither true or false")
    print("Sucesses:", len(successes))
    print("Fails:", len(fails))

    string_sample_1 = [json.dumps(json_og) for json_og in fails]
    fails_example =  [json.dumps(json_og["example"]) for json_og in fails]

    with open("spider_codex_k5_true.jsonl", "w") as out:
        out.write("\n".join(string_sample_1))
        
    with open("spider_codex_conversion_k_5_n_2691.jsonl") as f:
        sample_2 = list(f)
    sample_2 = [json.dumps(json.loads(json_str)["example"]) for json_str in sample_2]

    diff = set(fails_example) ^ set(sample_2)

    with open("spider_codex_diff.jsonl", "w") as out:
         out.write("\n".join(diff))

    