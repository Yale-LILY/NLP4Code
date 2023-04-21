import os
from typing import List, Dict, Tuple

NLP4CODE_TEST_DATA_PATH = os.environ["NLP4CODE_TEST_DATA_PATH"]


from execution.executors import MathExecutor
from finetuning.lightning_modules.datasets.base_reader import (
    NL2CodeDataModule,
    NL2CodeDataset,
)
from finetuning.lightning_modules.datasets.mathqa_reader import (
    FewShotMathQADataModule,
    FewShotMathQADataset,
    MathQADataset,
    MathQADataModule,
    MathQAEndVerificationDataset,
    MathQAEndVerificationDataModule,
)
from finetuning.lightning_modules.datasets.mbpp_reader import (
    FewShotMBPPDataModule,
    FewShotMBPPDataset,
    MBPPEndVerificationDataModule,
    MBPPEndVerificationDataset,
)
from finetuning.lightning_modules.datasets.spider_reader import (
    FewShotSpiderDataset,
    FewShotSQLDataModule,
    SpiderDataset,
    SpiderEndVerificationDataset,
    SQLEndVerificationDataModule,
    Text2SqlDataModule,
)

# TODO: use special test string for test transformer model name?
TEST_TRANSFORMER_MODEL_NAME = "EleutherAI/gpt-neo-125M"

# ======== datasets ========

# TODO: better way to do this? (custom types for each kwargs?)
# list of (dataset, **init_kwargs) tuples
FEW_SHOT_DATASETS: List[Tuple[NL2CodeDataset, Dict]] = [
    (
        FewShotMathQADataset,
        {
            "prompt_file": "prompt_files/mathqa_non_idiomatic_code_init_val.txt",
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/mathqa/val_dedup_init_val.jsonl",
            "mode": "test_few_shot",
        },
    ),
    (
        FewShotMBPPDataset,
        {
            "prompt_file": "prompt_files/mbpp_prompt_1_test.txt",
            "add_assertion_n": 1,
            "mode": "test_few_shot",
        },
    ),
    (
        FewShotSpiderDataset,
        {
            "prompt_file": "prompt_files/spider_codex_cot_sql_prompt_baseline_very_short.txt",
            "mode": "test_few_shot",
        },
    ),
]


DATASETS: List[Tuple[NL2CodeDataset, Dict]] = [
    (
        MathQADataset,
        {
            # "file_path": "data/mathqa/train-python.jsonl",
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/mathqa/train_dedup.jsonl",
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            # TODO: test different modes
            "mode": "train",
        },
    ),
    (
        SpiderDataset,
        {
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/spider/train_spider_processed_v2.jsonl",
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "mode": "train",
        },
    ),
]

# FEW_SHOT_DATA_MODULES: List[NL2CodeDataModule] = [
#     FewShotMathQADataModule,
#     FewShotMBPPDataModule,
#     FewShotSQLDataModule,
# ]

# train_file_path: data/mathqa/train-python.jsonl
# val_file_path: data/mathqa/val-python.jsonl
DATA_MODULES: List[Tuple[NL2CodeDataModule, Dict]] = [
    (
        MathQADataModule,
        {
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "train_set_init_args": {
                "file_path": f"{NLP4CODE_TEST_DATA_PATH}/mathqa/train_dedup.jsonl"
            },
            "val_set_init_args": {
                "file_path": f"{NLP4CODE_TEST_DATA_PATH}/mathqa/val_dedup.jsonl"
            },
        },
    ),
    (
        Text2SqlDataModule,
        {
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "train_set_init_args": {
                "file_path": f"{NLP4CODE_TEST_DATA_PATH}/spider/train_spider_processed_v2.jsonl"
            },
            "val_set_init_args": {
                "file_path": f"{NLP4CODE_TEST_DATA_PATH}/spider/dev_processed.jsonl"
            },
            "train_max_instances": 10,
            "val_max_instances": 10,
        },
    ),
]
