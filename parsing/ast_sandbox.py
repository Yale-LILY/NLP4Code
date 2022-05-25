from sql2pandas import sql2pandas
import ast
import json
from preprocess import check_processed_sql_tree, preprocess_sql_query_into_tree, sql_query_to_pandas_code_snippets
from tqdm import tqdm

f_train_spider_dataset = open('data/spider/train_spider.json')

# RESULTS:
# - JOIN does not generate formatting error but gives incorrect query
# - nested SELECTs are invalid (can be preprocessed)
# - INTERSECT/UNION/EXCEPT (can be preprocessed)
# - need LIKE '*%' to be single quotes: just convert all to single quotes at beginning


def clean_sql_query(query):
    query = query.replace('"', '\'')
    return query if query[len(query) - 1] == ';' else query + ';'


json_data = json.load(f_train_spider_dataset)
queries = []
for entry in json_data:
    query = entry['query']
    cleaned_query = clean_sql_query(query)

    entry['query'] = cleaned_query
    queries.append(cleaned_query)


bad_tokens = ['JOIN', 'INTERSECT', 'EXCEPT', 'UNION']
count_total_queries = len(queries)

bad_token_counts = {}
for bad_token in bad_tokens:
    bad_token_counts[bad_token] = 0

other_bad_queries = []
count_other_bad_queries = 0

count_total_bad_queries = 0


output_json = []

# with open('sandbox_queries_to_snippets.txt', 'w', encoding='utf-8') as f_write:
for idx, query in tqdm(enumerate(queries)):

    if not any([bad_token in query for bad_token in bad_tokens]):
        continue

    print(f"-------- {idx+1} --------\n")
    query = queries[idx]
    snippets = sql_query_to_pandas_code_snippets(query)
    snippets_str = "\n".join(snippets)
    print(f"{query}\n\n{snippets_str}\n\n\n")
