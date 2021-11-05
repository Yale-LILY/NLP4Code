from comparison_eval import bleu_score
import openai
import os
import datasets
import pandas as pd
import json
import pickle
from tqdm import tqdm


openai.api_key = os.environ.get('OPENAI_API_KEY')
# apps_dataset = pd.read_json('./data/apps/preprocessed/val.jsonl', lines=True)

with open('./data/apps/preprocessed/val.jsonl', 'r') as f:
    apps_dataset = [json.loads(line) for line in f]


# completions = []
# for app in tqdm(apps_dataset):
#     # print(app['question'])
#     completion = openai.Completion.create(engine="davinci-codex", prompt=app['question'], max_tokens=256)
#     completions.append(completion.choices[0].text)
#     # print(completion)
#     # print("Next completion:")

# with open('codexoutputs.pickle', 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(completions, f)

with open('codexoutputs.pickle', 'rb') as f:
    completions = pickle.load(f)

# print(apps_dataset[0]["solutions"][0]["raw_code"])

# print(bleu_score(apps_dataset[0]["solutions"][0]["raw_code"], completions[0]))

total_bleu = 0
i = 0
for app, completion in tqdm(zip(apps_dataset, completions)):
    if i != 196:
        total_bleu = total_bleu + bleu_score(app["solutions"][0]["raw_code"], completion)
    i = i + 1
    
print(total_bleu / len(apps_dataset))