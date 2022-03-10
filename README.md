# NLP4Code
Repository for the NLP4Code project at the LILY lab.

## Installation
[Optional] Create a virtualenv or conda enviroment  
Then, install the dependencies:
```
pip install -r requirements.txt
```

## Wandb
We use Wandb for experiment tracking. Please register ask Ansong for an invitation to the Wandb Yale-LILY team before 
running experiments. When you are ready to run the exps and log it to the cloud, do the following:
```
wandb login
```
Paste your API key and the login is complete. When start running experiments, you should see something like 
```
wandb: Tracking run with wandb version 0.12.11
wandb: Run data is saved locally in /home/ansongni/Code/NLP4Code/wandb/run-20220309_150158-1ebacxm4
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run mathqa-gpt-finetuning
wandb: ⭐️ View project at https://wandb.ai/yale-lily/unified-codegen
wandb: 🚀 View run at https://wandb.ai/yale-lily/unified-codegen/runs/1ebacxm4
```

If you want to do some test runs without logging to the cloud, run `wandb offline` first as suggested above. 

## Fine-tuning
(Read the previous section first if you are ready to run experiments)
For fine-tuning, in the main directory, do:
```
export PYTHONPATH=`pwd`; python finetuning/trainer.py fit --config finetuning/training_configs/*.yaml
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

## something else
