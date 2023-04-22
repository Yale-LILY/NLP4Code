import os
import unittest

# get the data directory from the environment variable
DATA_DIR = os.environ.get('DATA_DIR')

class TestDecOnlyModelInference(unittest.TestCase):
    def test_basic(self):
        exit_code = os.system("export PYTHONPATH=`pwd`; " + \
                                "python finetuning/trainer.py validate " + \
                                "--config tests/test_configs/few_shot-pipeline.yaml " + \
                                # "--trainer.gpus 0 " + \
                                # "--trainer.accelerator cpu " + \
                                # "--trainer.precision 32 " + \
                                "--model.init_args.transformer_model_name EleutherAI/gpt-neo-125M " + \
                                "--data.init_args.transformer_model_name EleutherAI/gpt-neo-125M " + \
                                "--data.init_args.val_max_instances 2 " + \
                                # "--data.init_args.val_batch_size 1 ")
                                "--data.init_args.val_batch_size 1 " + \
                                "--data.init_args.val_set_init_args.file_path $NLP4CODE_TEST_DATA_PATH/gsmath/split_dev.jsonl ")
        
        self.assertEqual(exit_code, 0)

if __name__ == '__main__':
    unittest.main()