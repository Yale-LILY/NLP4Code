import json
from collections import Counter
import pandas as pd

SQL_KEYWORDS = [
"ADD", 
"ADD CONSTRAINT", 
"ALL",
"ALTER",
"ALTER COLUMN",
"ALTER TABLE",
"AND",
"ANY",
"AS",
"ASC",
"BACKUP DATABASE",
"BETWEEN",
"CASE",
"CHECK",
"COLUMN",
"CONSTRAINT",
"DATABASE",
"DEFAULT",
"DELETE",
"DESC",
"DISTINCT",
"EXEC",
"EXISTS",
"FOREIGN KEY",
"FROM",
"FULL OUTER JOIN",
"GROUP BY",
"HAVING",
"IN",
"INDEX",
"INNER JOIN",
"INSERT INTO",
"INSERT INTO SELECT",
"IS NULL",
"IS NOT NULL",
"JOIN",
"LEFT JOIN",
"LIKE",
"LIMIT",
"NOT",
"NOT NULL",
"OR",
"ORDER BY",
"OUTER JOIN",
"PRIMARY KEY",
"PROCEDURE",
"RIGHT JOIN",
"ROWNUM",
"SELECT",
"SELECT DISTINCT",
"SELECT INTO",
"SELECT TOP",
"SET",
"TABLE",
"TOP",
"TRUNCATE TABLE",
"UNION",
"UNION ALL",
"UNIQUE",
"UPDATE",
"VALUES",
"VIEW", 
"WHERE", # BELOW MY ADDONS TO W3SCHOOLS
"COUNT(*)",
"AVG",
"SUM",
"MAX",
"MIN",
"INTERSECT",
"EXCEPT",
"(SELECT"
]

if __name__ == "__main__":
    sample_1 = []
    sample_5 = []
    sample_20 = []

    with open("spider_codex_conversion_k_1_n_6997.jsonl") as f:
        for line in f:
            sample_1.append(json.loads(line))
    with open("spider_codex_conversion_k_5_n_2691.jsonl") as f:
        for line in f:
            sample_5.append(json.loads(line))
    with open("spider_codex_conversion_k_20_n_1969.jsonl") as f:
        for line in f:
            sample_20.append(json.loads(line))

    data = [sample_1, sample_5, sample_20]

    # Keep set of all tokens 
    # all_toks = set()
    # for item in sample_1:
    #     for tok in item["example"]["query_toks"]:
    #         all_toks.add(tok)
    
    fail = []

    for data_set in data:
        print("Num. Examples:", len(data_set))
        successes = []
        fails = []
        for item in data_set:
            truthy_results = [x[1] for x in item["program_result_list"]]
            if True in truthy_results:
                successes.append(item)
            elif False in truthy_results:
                fails.append(item)
            else:
                raise Exception("Result neither true or false")
        print("Sucesses:", len(successes))
        print("Fails:", len(fails))

        # Add fails to fail list
        fail.append(fails)
    
    assert(len(fail) == 3)

    fail_errs = []

    for lst in fail:
        i = 0
        err = Counter(SQL_KEYWORDS)
        err.subtract(err) # Reset counts to 0

        for item in lst:
            for tok in SQL_KEYWORDS:
                if tok.lower() in item["example"]["query"].lower():
                    err[tok] += 1
            if i % 100 == 0:
                print(i)
            i += 1
        
        fail_errs.append(err)

    print("Sample 1:")
    for k, v in fail_errs[0].most_common():
        print(k, v)
    print("\n")

    print("Sample 5:")
    for k, v in fail_errs[1].most_common():
        print(k, v)
    print("\n")

    print("Sample 20:")
    for k, v in fail_errs[2].most_common():
        print(k, v)
    print("\n")

        

