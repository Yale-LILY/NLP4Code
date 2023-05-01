import os
from typing import List, Dict, Tuple, Optional

NLP4CODE_TEST_DATA_PATH = os.environ["NLP4CODE_TEST_DATA_PATH"]


from finetuning.lightning_modules.datasets.base_reader import (
    FewShotNL2CodeDataset,
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

from execution.executors import (
    BaseExecutor,
    MathExecutor,
    MBPPExecutor,
    SpiderExecutor,
    WTQExecutor,
)


# TODO: use special test string for test transformer model name? (don't load model)
TEST_TRANSFORMER_MODEL_NAME = "EleutherAI/gpt-neo-125M"


# ======== datasets ========

# defines kwargs needed to initialize NL2CodeDataset
class TestDatasetInitKwargs:
    transformer_model_name: str
    file_path: str
    mode: str

    def __init__(
        self,
        file_path: str,
        mode: Optional[str] = "train",  # default to train
        transformer_model_name: Optional[str] = TEST_TRANSFORMER_MODEL_NAME,
    ):
        self.file_path = file_path
        self.mode = mode
        self.transformer_model_name = transformer_model_name


DATASETS: List[Tuple[NL2CodeDataset, TestDatasetInitKwargs]] = [
    (
        MathQADataset,
        TestDatasetInitKwargs(
            file_path=f"{NLP4CODE_TEST_DATA_PATH}/mathqa/train_dedup.jsonl",
        ),
    ),
    # TODO: SpiderDataset prompt_function
    # (
    #     SpiderDataset,
    #     TestDatasetInitKwargs(
    #         file_path=f"{NLP4CODE_TEST_DATA_PATH}/spider/train_spider_processed_v2.jsonl",
    #     ),
    # ),
]


# defines kwargs needed to instantiate FewShotNL2CodeDataset
class TestFewShotDatasetInitKwargs(TestDatasetInitKwargs):
    transformer_model_name: str
    file_path: str
    exemplar_file_path: str
    mode: str = "test"

    def __init__(
        self,
        file_path: str,
        exemplar_file_path: str,
        transformer_model_name: Optional[str] = TEST_TRANSFORMER_MODEL_NAME,
    ):
        super().__init__(
            file_path=file_path,
            transformer_model_name=transformer_model_name,
            mode="test",
        )
        self.exemplar_file_path = exemplar_file_path


# TODO: better way to do this? (custom types for each kwargs?)
# TODO: make sure to keep dataset files up to date here
# list of (dataset, **init_kwargs) tuples
FEW_SHOT_DATASETS: List[Tuple[FewShotNL2CodeDataset, TestFewShotDatasetInitKwargs]] = [
    (
        FewShotMathQADataset,
        TestFewShotDatasetInitKwargs(
            exemplar_file_path="prompt_files/mathqa-non_idiomatic_code-annotated-8_exemplars.jsonl",
            file_path=f"{NLP4CODE_TEST_DATA_PATH}/mathqa/val_dedup_init_val.jsonl",
        ),
    ),
    (
        FewShotMBPPDataset,
        TestFewShotDatasetInitKwargs(
            exemplar_file_path="prompt_files/mbpp-official_first_3-10_exemplars.jsonl",
            file_path=f"{NLP4CODE_TEST_DATA_PATH}/mbpp/mbpp_test.jsonl",
        ),
    ),
    (
        FewShotSpiderDataset,
        TestFewShotDatasetInitKwargs(
            exemplar_file_path="prompt_files/spider-8_exemplars.jsonl",
            file_path=f"{NLP4CODE_TEST_DATA_PATH}/spider/dev_processed_db_path.jsonl",
        ),
    ),
    (
        FewShotSpiderDataset,
        TestFewShotDatasetInitKwargs(
            exemplar_file_path="prompt_files/wtq-8_exemplars.jsonl",
            # TODO: why does wtq_restored_dev.jsonl error
            file_path=f"{NLP4CODE_TEST_DATA_PATH}/squall/wtq_restored_test.jsonl",
        ),
    ),
]


# ======== models ========

TEST_MODEL_TRANSFORMER_MODEL_NAMES: List[str] = [
    "EleutherAI/gpt-neo-125M",
    "Salesforce/codet5-small",
    "Salesforce/codegen-350M-multi",
]

TEST_MODEL_EXECUTOR_CLS = "execution.executors.MathExecutor"


# ======== executors ========

TEST_PROGRAM = "answer = 5"

# Tuple[ExecutorCls, program, example]
TEST_EXECUTORS: List[Tuple[BaseExecutor, str, Dict]] = [
    (
        MathExecutor,
        TEST_PROGRAM,
        {
            "question": "some question",
            "answer": 5,
        },
    ),
    (
        MBPPExecutor,
        TEST_PROGRAM,
        {
            "question": "some question",
            "answer": 5,
            "code": "return 5",
            "task_id": "xyz",
            "test_setup_code": 'print("setup")',
            "test_list": ["assert 1+1 == 2", "assert 1+1 != 3"],
        },
    ),
    (
        SpiderExecutor,
        TEST_PROGRAM,
        {
            "question": "some question",
            "db_id": "my_db_id",
            "query": "SELECT * FROM table",
        },
    ),
    (
        WTQExecutor,
        TEST_PROGRAM,
        {
            "question": "some question",
            "db_id": "my_db_id",
            "db_path": "path/to/my/db",
            "original_answer": 5,
        },
    ),
]
