from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
from lightning_modules.models.seq2seq_model_util import get_model
from lightning_modules.datasets.spider_reader import SpiderDataset, Text2SqlDataModule


MODEL_NAME = "EleutherAI/gpt-neo-125M"

model, tokenizer = get_model(MODEL_NAME, gradient_ckpt=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="out",
    gradient_checkpointing=True,
    learning_rate=5e-05,
    weight_decay=0.01,
    max_steps=25000,
    fp16=True,
    do_train=True,
    # do_eval=True,
    logging_strategy="no",
)

dataset_init_args = {
    "transformer_model_name": MODEL_NAME,
    # "batch_size": 4,
    # "val_batch_size": 4,
    # "train_max_instances": 200,
    # "val_max_instances": 100,
    "file_path": "data/spider/train_spider_processed_v2.jsonl",
}

# dataset = Text2SqlDataModule(
#     transformer_model_name=MODEL_NAME,
#     batch_size=4,
#     val_batch_size=4,
#     train_max_instances=200,
#     val_max_instances=100,
#     train_set_init_args={
#         "file_path": "data/spider/train_spider_processed_v2.jsonl",
#     },
#     val_set_init_args={
#         "file_path": "data/spider/dev_processed.jsonl",
#     },
#     set_common_init_args={
#         "use_skg_format": False,
#     },
# )

dataset = SpiderDataset(
    transformer_model_name=MODEL_NAME,
    file_path="data/spider/train_spider_processed_v2.jsonl",
)

# print(dataset)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
