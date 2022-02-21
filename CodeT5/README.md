## Updates
Please read ORIG_README.md first and set up environment following its instruction
### Feb 20, 2022
1. Change run_gen.py From DataParallel to DistributedDataParallel
2. Preprocssing Spider dataset (in /preprocessing_spider)
3. Set up finetuning pipeline and try on Spider dataset
4. Set up inference notebook (in CodeT5/codet5_spider_inference.ipynb)
5. [TODO] try official evaluation scipt of Spider and post the evaluation score


Input format of Spider (also for other dataset/benchmark added in the future):
```
[
    {
        "intent": "How many heads of the departments are older than 56 ? | department_management | department : Department_ID ( Kyle ), Name, Creation, Ranking, Budget_in_Billions, Num_Employees | head : head_ID ( Kyle ), name, born_state, age | management : department_ID ( Kyle ), head_ID, temporary_acting",
        "snippet": "SELECT count(*) FROM head WHERE age  >  56"
    },
    ...
}
```

Command to finetune Spider
```python
CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=. python -m torch.distributed.launch run_gen.py --task spider --num_train_epochs 30 --cache_path cache_dir --summary_dir summary_dir --data_dir data --res_dir result_dir --output_dir outputs --model_name_or_path Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --do_train --do_eval --save_last_checkpoints --patience 20
```
Note: 
- Some directories (e.g. outputs) are not uploaded to the github repo, and need to be created locally at first
- Check the file path, some might be used as relative path
