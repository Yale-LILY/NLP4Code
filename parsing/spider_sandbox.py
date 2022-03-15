from sql2pandas import sql2pandas
import ast
import json

f_train_spider_dataset = open(
    '/Users/TFunk/Downloads/spider/train_spider.json')

# RESULTS:
# - JOIN does not generate formatting error but gives incorrect query
# - nested SELECTs are invalid (can be preprocessed)
# - INTERSECT/UNION/EXCEPT (can be preprocessed)
# - need LIKE '*%' to be single quotes: just convert all to single quotes at beginning


def preprocess_sql_query(query):
    query = query.replace('"', '\'')
    return query if query[len(query) - 1] == ';' else query + ';'


json_data = json.load(f_train_spider_dataset)
queries = []
for entry in json_data:
    query = entry['query']
    queries.append(preprocess_sql_query(query))


bad_tokens = ['JOIN', 'INTERSECT', 'EXCEPT', 'UNION']
count_total_queries = len(queries)

bad_token_counts = {}
for bad_token in bad_tokens:
    bad_token_counts[bad_token] = 0

other_bad_queries = []
count_other_bad_queries = 0

count_total_bad_queries = 0


output_json = []

for idx, query in enumerate(queries):
    # for idx in range(100):
    #     query = queries[idx]
    if idx % 100 == 0:
        print(idx)
    # for i in range(100):
    #     query = queries[i]
    should_skip_output_json = False
    for sql_token in bad_tokens:
        if query.find(sql_token) >= 0:
            bad_token_counts[sql_token] += 1
            # bad_queries.append(query)
            should_skip_output_json = True

    converted_query = sql2pandas(query)
    if converted_query.find('Error:') >= 0:
        count_other_bad_queries += 1
        other_bad_queries.append(query)
        should_skip_output_json = True

    if should_skip_output_json == True:
        count_total_bad_queries += 1
        continue

    full_json_entry = json_data[idx]
    full_json_entry['pandas_converted'] = converted_query
    output_json.append(full_json_entry)

for bad_query in other_bad_queries:
    print(bad_query)

for key, value in bad_token_counts.items():
    print(key + ': ' + str(value))
print('Other bad queries: ' + str(count_other_bad_queries))
print('Total bad queries: ' + str(count_total_bad_queries))
print('Total queries: ' + str(count_total_queries))

with open('train_spider_converted.json', 'w', encoding='utf-8') as f_write:
    json.dump(output_json, f_write, ensure_ascii=False, indent=4)
