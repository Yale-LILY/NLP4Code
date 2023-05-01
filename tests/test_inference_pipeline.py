import os
import unittest

# get the data directory from the environment variable
DATA_DIR = os.environ.get("DATA_DIR")

from tests.consts import TEST_PIPELINE_INFO


class TestDecOnlyModelInference(unittest.TestCase):
    def test_basic(self):
        for model_name, yaml_config_path, val_file_path in TEST_PIPELINE_INFO:
            exit_code = os.system(
                "export PYTHONPATH=`pwd`; echo $PYTHONPATH; echo $NLP4CODE_TEST_DATA_PATH; "
                + "python finetuning/trainer.py validate "
                + f"--config {yaml_config_path} "
                + "--trainer.gpus 0 "  # still using CPU for now
                + "--trainer.accelerator cpu "
                + "--trainer.precision 32 "
                + "--model.init_args.print_eval_every_n_batches 1 "
                + f"--model.init_args.transformer_model_name {model_name} "
                + f"--data.init_args.transformer_model_name {model_name} "
                + "--data.init_args.val_max_instances 2 "
                + "--data.init_args.val_batch_size 1 "  # "--data.init_args.val_batch_size 1 ")
                + f"--data.init_args.val_file_path {val_file_path} "
            )

            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
