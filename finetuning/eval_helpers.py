from torchmetrics import Metric, MeanMetric, MetricCollection
from transformers.trainer_utils import EvalPrediction
from typing import Any, Dict, List

from execution.executors import BaseExecutor, SpiderExecutor
from lightning_modules.datasets.spider_reader import SpiderDataset, Text2SqlDataModule
from lightning_modules.models.seq2seq_model import Seq2SeqModel

from consts import MODEL_NAME, MAX_STEPS


spider_data_module = Text2SqlDataModule(
    transformer_model_name=MODEL_NAME,
    batch_size=4,
    val_batch_size=4,
    train_max_instances=200,
    val_max_instances=100,
    train_set_init_args={"file_path": "data/spider/train_spider_processed_v2.jsonl"},
    val_set_init_args={
        "file_path": "data/spider/dev_processed.jsonl",
    },
    set_common_init_args={
        "use_skg_format": False,
    },
)

spider_data_module.setup(stage="fit")

train_dataset = spider_data_module.train_data

eval_dataset = spider_data_module.val_data

seq2seq_model = Seq2SeqModel(
    transformer_model_name=MODEL_NAME,
    gradient_ckpt=True,
    executor_cls="execution.executors.SpiderExecutor",
    categorize_func="execution.spider_execution.spider_categorize_complexity",
    category_list=["JOIN", "NESTED", "COMPOUND", "SIMPLE"],
    max_gen_len=128,
    sampling_temp=0.01,
    optimizer={
        "init_args": {
            "lr": 5.0e-5,
            # lr: 0.0,
            "betas": [0.9, 0.99],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        }
    },
    lr_scheduler={
        "name": "linear",
        "init_args": {
            "num_warmup_steps": 100,
            "num_training_steps": MAX_STEPS,
        },
    },
)
seq2seq_model.model = seq2seq_model.model.cuda()
# seq2seq_model.model.config.max_new_tokens = 1024
# print(seq2seq_model.model.config.max_new_tokens)
# seq2seq_model.model.config.max_length = 1024

executor = SpiderExecutor()


def get_program_exec_dict(
    generated_program: str, exec_match: int, exec_result: Any
) -> Dict[str, Any]:
    exec_acc = 1.0 if exec_match == 1 else 0.0
    exec_rate = 0.0 if exec_match == -1 else 1.0

    # save the results in the json output file
    save_metrics = {"exec_acc": float(exec_acc), "exec_rate": float(exec_rate)}

    # add more information to the program dict
    program_dict = {"program": generated_program, "exec_result": exec_result}
    program_dict.update(save_metrics)

    return program_dict


val_instances = eval_dataset.instances


def validation_step_end(
    eval_pred: EvalPrediction,
    metrics_dict: Dict[str, Metric],
    executor: BaseExecutor,
    val_instances: List[Any],
) -> None:
    n = len(val_instances)
    # update the evaluation metrics
    for i in range(n):
        prediction = eval_pred.predictions[1][i]
        label_id = eval_pred.label_ids[i]
        print(list(prediction))
        print(list(seq2seq_model.tokenizer.convert_tokens_to_ids(prediction)))
        print(seq2seq_model.tokenizer.decode(prediction))
        print(seq2seq_model.tokenizer.decode(label_id))
        # example = eval_pred.label_ids[i]
        # example = eval_pred.inputs[i]
        example = val_instances[i]["metadata"]

        # obtain the execution results
        exec_match, exec_result = executor.exec_program(prediction, example)
        program_len_diff = executor.program_len(prediction) - executor.gold_program_len(
            example
        )
        program_dict = get_program_exec_dict(prediction, exec_match, exec_result)

        # update the metrics
        metrics_dict["exec_acc"](program_dict["exec_acc"])
        metrics_dict["exec_rate"](program_dict["exec_rate"])
        metrics_dict["program_len_diff"](program_len_diff)
        # category_metrics.update(program_dict["exec_acc"], metadata) # note that this can't be forward as compute will be called

    # if print_eval_every_n_batches > 0:
    #     # compute the metrics
    #     eval_metrics_dict = {}
    #     for k in metrics_dict.keys():
    #         eval_metrics_dict[k] = float(metrics_dict[k].compute())
    #     print("eval metrics: ", eval_metrics_dict)

    # # save the outputs to the model
    # predictions.extend(outputs)


# attempt to use compute_metrics to inject custom validation
def compute_metrics(eval_pred: EvalPrediction) -> dict:
    print(len(eval_pred.predictions))
    print(eval_pred.predictions[0].shape)
    print(eval_pred.predictions[0])
    print(eval_pred.predictions[1].shape)
    print(eval_pred.predictions[1])
    print("\n========\n")
    print(len(eval_pred.label_ids))
    print(eval_pred.label_ids[0].shape)
    print(eval_pred.label_ids[0])
    print(eval_pred.label_ids[1].shape)
    print(eval_pred.label_ids[1])
    # # n = len(eval_pred.predictions)
    metrics_dict: Dict[str, Metric] = MetricCollection({})
    metrics_dict["exec_acc"] = MeanMetric()
    metrics_dict["exec_rate"] = MeanMetric()
    metrics_dict["program_len_diff"] = MeanMetric()
    # metrics_dict = {}
    # metrics_dict["exec_acc"] = 0.0
    # metrics_dict["exec_rate"] = 0.0
    # metrics_dict["program_len_diff"] = 0.0
    # print(eval_pred.predictions)

    validation_step_end(
        eval_pred=eval_pred,
        metrics_dict=metrics_dict,
        executor=executor,
        val_instances=val_instances,
    )
    # print("TEST" + str(eval_pred))
    print(metrics_dict)
    return metrics_dict
