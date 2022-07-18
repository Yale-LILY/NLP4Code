import json

from typing import List, Dict, Any, Callable

correct_fn = lambda x: float(any([y[1] for y in x['program_result_list']])) == 1.0
incorrect_fn = lambda x: float(all([y[1] == False for y in x['program_result_list']])) == 1.0

has_join = lambda x: 'join' in x['example']['query'].lower()

has_complex = lambda x: 'intersect' in x['example']['query'].lower() or \
                        'union' in x['example']['query'].lower() or \
                        'except' in x['example']['query'].lower() or \
                        '(select' in x['example']['query'].lower()

has_group_by = lambda x: 'group by' in x['example']['query'].lower()

has_order_by = lambda x: 'order by' in x['example']['query'].lower()

def separate_results(fn: Callable, name: str, results: List[Dict[str, Any]]):
    n_total = len(results)
    n_total_correct = len(list(filter(correct_fn, results)))

    with_attr_results = list(filter(fn, results))
    with_attr_correct_results = list(filter(correct_fn, with_attr_results))

    print(f"with {name} ({len(with_attr_results)/n_total:.2%}), " + \
          f"success/total: {len(with_attr_correct_results)}/{len(with_attr_results)}, " + \
          f"rate is {len(with_attr_correct_results)/len(with_attr_results):.2%}")
    # print(f"without {name}, success/total: {n_total_correct - len(with_attr_correct_results)}/{n_total - len(with_attr_results)}, rate is {(n_total_correct - len(with_attr_correct_results))/(n_total - len(with_attr_results)):.2%}")

def sql_type_error_analysis(results: List[Dict[str, Any]]):
    # separate_results(correct_fn, 'correct', results)
    separate_results(has_join, 'join', results)
    separate_results(has_complex, 'intersect/union/except/nested', results)
    separate_results(has_group_by, 'group by', results)
    separate_results(has_order_by, 'order by', results)

    separate_results(lambda x: has_join(x) or has_complex(x) or has_group_by(x), 'any complex things', results)


def main():
    with open('few_shot_results/spider_codex_conversion_k_20_n_1969.jsonl', 'r') as f:
        results = [json.loads(s) for s in f.readlines()]

    results = list(filter(incorrect_fn, results))
    sql_type_error_analysis(results)
    

if __name__ == '__main__':
    main()