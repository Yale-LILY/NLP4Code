import json
import pandas as pd

# Set this to the absolute path to the squall.json file
SQUALL_PATH = "/mnt/c/Users/Stephen Yin/Desktop/14th Grade/Research Lab/squall.json"

with open(SQUALL_PATH, "r") as read_file:
    json_data = json.load(read_file)

# Tracker variables
n = len(json_data)
n_cols = len(json_data[0])

print("There are " + str(n) + " number of entries in Squall.")
print("There are " + str(n_cols) + " number of columns per entry in Squall.")

# See the columns
df = pd.DataFrame(json_data)
df.info()

# We see that there are two extra columsn compared to the documentation: 
#     nl_typebio, nl_typebio_col 
# Let's examine what they are:
print(type(df["nl_typebio"][0]))
print(type(df["nl_typebio_col"][0]))
print(type(df["nl_typebio"][0][0]))
print(type(df["nl_typebio_col"][0][5]))
print(df["nl_typebio"].head(5))
print(df["nl_typebio_col"].head(5))

# They are both lists of STRINGS/NONE that _seem_ to be describing some tagging data on the sentence
"""
Begin: Preprocessing squall.json per entry, following format of spider_sandbox.py written by Troy
"""
def preprocess_sql_query(query):

    # Get only SQL words
    words = []
    for word in query:
        words.append(word[1])
    
    query = " ".join(words)
    query = query.replace('"', '\'')
    print(query)
    
    return query if query[len(query) - 1] == ';' else query + ';'

queries = []
for entry in json_data[:2]:
    query = entry['sql']
    queries.append(preprocess_sql_query(query))

# bad_tokens = ['JOIN', 'INTERSECT', 'EXCEPT', 'UNION']
# count_total_queries = len(queries)

# bad_token_counts = {}
# for bad_token in bad_tokens:
#     bad_token_counts[bad_token] = 0

# other_bad_queries = []
# count_other_bad_queries = 0

# count_total_bad_queries = 0


# output_json = []

# for idx, query in enumerate(queries):
#     # for idx in range(100):
#     #     query = queries[idx]
#     if idx % 100 == 0:
#         print(idx)
#     # for i in range(100):
#     #     query = queries[i]
#     should_skip_output_json = False
#     for sql_token in bad_tokens:
#         if query.find(sql_token) >= 0:
#             bad_token_counts[sql_token] += 1
#             # bad_queries.append(query)
#             should_skip_output_json = True

#     converted_query = sql2pandas(query)
#     if converted_query.find('Error:') >= 0:
#         count_other_bad_queries += 1
#         other_bad_queries.append(query)
#         should_skip_output_json = True

#     if should_skip_output_json == True:
#         count_total_bad_queries += 1
#         continue

#     full_json_entry = json_data[idx]
#     full_json_entry['pandas_converted'] = converted_query
#     output_json.append(full_json_entry)

# for bad_query in other_bad_queries:
#     print(bad_query)

# for key, value in bad_token_counts.items():
#     print(key + ': ' + str(value))
# print('Other bad queries: ' + str(count_other_bad_queries))
# print('Total bad queries: ' + str(count_total_bad_queries))
# print('Total queries: ' + str(count_total_queries))

# with open('squall_converted.json', 'w', encoding='utf-8') as f_write:
#     json.dump(output_json, f_write, ensure_ascii=False, indent=4)
