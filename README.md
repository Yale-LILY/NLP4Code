# NLP4Code
Repository for the NLP4Code project at the LILY lab.

## Installation
[Optional] Create a virtualenv or conda enviroment  
Then, install the dependencies:
```
pip install -r requirements.txt
```

## Fine-tuning
For fine-tuning, in the main directory, do:
```
export PYTHONPATH=`pwd`; python finetuning/trainer.py fit --config finetuning/training_configs/mathqa_gpt_finetuning.yaml
```

## Preprocessing APPS
create the following directory and uncompress the original data
```
mkdir -p data/apps
cd data/apps
wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
tar -cvzf APPS.tar.gz
```
Then run the preprocessing script
```
python preprocessing_apps.py
```
After this step, you should see the train/val/test files in `data/apps/preprocessing/[train, val, test].jsonl`

For most tasks (*e.g.,* evaluating perplexity), you can simply use `example['solutions']['raw_code']`

## Evaluation
Currently the following evaluation metrics are implemented:
* Reference code needed:
    * BLEU
    * ROUGE
* Reference code not needed:
    * Parsability
    * Statment count
    * Perplexity
