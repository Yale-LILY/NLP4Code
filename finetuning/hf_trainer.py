from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import EvalPrediction, IntervalStrategy
from lightning_modules.models.seq2seq_model import Seq2SeqModel
from lightning_modules.models.seq2seq_model_util import get_model
from lightning_modules.datasets.spider_reader import SpiderDataset, Text2SqlDataModule

from eval_helpers import (
    compute_metrics,
    seq2seq_model,
    train_dataset,
    eval_dataset,
    spider_data_module,
)

from lightning_modules.datasets.base_reader import (
    customized_collate_fn_enc_dec,
    customized_collate_fn_gpt,
)
from finetuning.lightning_modules.models.seq2seq_model_util import (
    is_model_gpt_style,
    right_pad_sequences,
)


import os
import torch
from typing import Dict


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "codegen-hf-migration-tests"

from consts import MODEL_NAME, MAX_STEPS, RUN_NAME, EVAL_STEPS

# model, tokenizer = get_model(
#     MODEL_NAME,
#     gradient_ckpt=True,
#     additional_init_args={
#         "executor_cls": "execution.executors.SpiderExecutor",
#         "categorize_func": "execution.spider_execution.spider_categorize_complexity",
#         "category_list": ["JOIN", "NESTED", "COMPOUND", "SIMPLE"],
#         "max_gen_len": 128,
#         "sampling_temp": 0.01,
#     },
# )

training_args = Seq2SeqTrainingArguments(
    output_dir="results/debug-tmp",  # local output dir
    do_train=True,
    do_eval=True,
    run_name=RUN_NAME,
    report_to="wandb",
    # hyperparams
    learning_rate=5e-05,
    weight_decay=0.01,
    max_steps=MAX_STEPS,
    fp16=True,
    # find batch size automatically to avoid cuda OOM: only compatible with accelerate
    auto_find_batch_size=True,
    # checkpointing
    save_strategy="epoch",
    # validation (?)
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=EVAL_STEPS,
    # per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    # memory optimizations
    gradient_checkpointing=True,
    ddp_find_unused_parameters=True,
    # deepspeed="deepspeed_config.json",
    predict_with_generate=True,
    generation_max_length=128,
)

collator = DataCollatorForSeq2Seq(
    tokenizer=seq2seq_model.tokenizer, model=seq2seq_model.model
)


class ValidationCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        print("======== Evaluation complete ========")
        print(state)

        # eval_dataloader = DataLoader(eval_dataset.val_data, batch_size=eval_dataset.val_batch_size,
        #                        shuffle=False, drop_last=True, collate_fn=collate_fn)
        val_outs = []
        eval_dataloader = spider_data_module.val_dataloader()
        for val_idx, val_batch in enumerate(iter(eval_dataloader)):
            val_batch["input_ids"] = val_batch["input_ids"].cuda()
            val_batch["attention_mask"] = val_batch["attention_mask"].cuda()
            # print(val_batch["metadata"])
            # print(val_batch["input_ids"].device)
            # print(val_batch["attention_mask"].device)
            out = seq2seq_model.validation_step(val_batch, val_idx)
            # print(out)
            # val_outs.append(out)
            val_outs.extend(out)
        seq2seq_model.validation_step_end(val_outs)
        # return super().on_evaluate(args, state, control, **kwargs)


class CustomTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        seq2seq_model: Seq2SeqModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ):
        self.seq2seq_model = seq2seq_model
        collator = DataCollatorForSeq2Seq(
            tokenizer=seq2seq_model.tokenizer, model=seq2seq_model.model
        )
        self.collator = collator
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        super().__init__(
            model=seq2seq_model.model,
            data_collator=collator,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=seq2seq_model.tokenizer,
            compute_metrics=compute_metrics,
        )

    def evaluate(self, eval_dataset, ignore_keys, metric_key_prefix):
        print("TEST")
        super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.seq2seq_model.training_step(batch=batch, batch_idx=batch_idx)


# trainer = CustomTrainer(
#     seq2seq_model=seq2seq_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
# )


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


trainer = Seq2SeqTrainer(
    model=seq2seq_model.model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
    callbacks=[ValidationCallback],
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    tokenizer=seq2seq_model.tokenizer,
)

# res = trainer.predict(eval_dataset[0:16])
# print(res)
# print(res.predictions)
# print(res.label_ids)
# decode_test = seq2seq_model.tokenizer.decode(res.label_ids[0])
# print(decode_test)

# trainer.evaluate()
trainer.train()
