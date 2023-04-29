import os
from typing import List, Dict, Tuple, Optional

NLP4CODE_TEST_DATA_PATH = os.environ["NLP4CODE_TEST_DATA_PATH"]


from finetuning.lightning_modules.datasets.base_reader import (
    NL2CodeDataset,
)
from finetuning.lightning_modules.datasets.mathqa_reader import (
    FewShotMathQADataset,
    MathQADataset,
)
from finetuning.lightning_modules.datasets.mbpp_reader import (
    FewShotMBPPDataset,
)
from finetuning.lightning_modules.datasets.spider_reader import (
    FewShotSpiderDataset,
    SpiderDataset,
)

# TODO: use special test string for test transformer model name? (don't load model)
TEST_TRANSFORMER_MODEL_NAME = "EleutherAI/gpt-neo-125M"

# ======== datasets ========


class TestFewShotDatasetInitKwargs:
    exemplar_file_path: str
    transformer_model_name: str
    file_path: str
    mode: str = "test"

    def __init__(
        self,
        exemplar_file_path: str,
        file_path: str,
        transformer_model_name: Optional[str] = TEST_TRANSFORMER_MODEL_NAME,
    ):
        self.exemplar_file_path = exemplar_file_path
        self.file_path = file_path
        self.transformer_model_name = transformer_model_name


# TODO: better way to do this? (custom types for each kwargs?)
# TODO: make sure to keep dataset files up to date here
# list of (dataset, **init_kwargs) tuples
FEW_SHOT_DATASETS: List[Tuple[NL2CodeDataset, Dict]] = [
    (
        FewShotMathQADataset,
        {
            "exemplar_file_path": "prompt_files/mathqa-non_idiomatic_code-annotated-8_exemplars.jsonl",
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/mathqa/val_dedup_init_val.jsonl",
            "mode": "test",
        },
    ),
    (
        FewShotMBPPDataset,
        {
            "exemplar_file_path": "prompt_files/mbpp-official_first_3-10_exemplars.jsonl",
            # "add_assertion_n": 1,
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/mbpp/mbpp_test.jsonl",
            "mode": "test",
        },
    ),
    (
        FewShotSpiderDataset,
        {
            "exemplar_file_path": "prompt_files/spider-8_exemplars.jsonl",
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/spider/dev_processed_db_path.jsonl",
            "mode": "test",
        },
    ),
    (
        FewShotSpiderDataset,
        {
            "exemplar_file_path": "prompt_files/wtq-8_exemplars.jsonl",
            "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
            # TODO: why does wtq_restored_dev.jsonl error
            "file_path": f"{NLP4CODE_TEST_DATA_PATH}/squall/wtq_restored_test.jsonl",
            "mode": "test",
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
    # TODO: SpiderDataset prompt_function
    # (
    #     SpiderDataset,
    #     {
    #         "file_path": f"{NLP4CODE_TEST_DATA_PATH}/spider/train_spider_processed_v2.jsonl",
    #         "transformer_model_name": TEST_TRANSFORMER_MODEL_NAME,
    #         "mode": "train",
    #     },
    # ),
]

# ======== models ========

TEST_MODEL_TRANSFORMER_MODEL_NAMES = [
    "EleutherAI/gpt-neo-125M",
    "Salesforce/codet5-small",
    "Salesforce/codegen-350M-multi",
]

TEST_MODEL_EXECUTOR_CLS = "execution.executors.MathExecutor"
