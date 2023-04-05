import unittest

from os import path

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))

from execution.executors import MathExecutor

from finetuning.lightning_modules.models.seq2seq_model import Seq2SeqModel


class TestModels(unittest.TestCase):
    def test_gpt_neo(self):
        model = Seq2SeqModel(
            transformer_model_name="EleutherAI/gpt-neo-125M",
            executor_cls="execution.executors.MathExecutor",
        )

        test_input_str = [
            "# write a python program that adds two integers",
            "# write a python program that adds two integers",
        ]
        context_tokenizer_outputs = model.tokenizer(test_input_str, return_tensors="pt")
        input_ids = context_tokenizer_outputs["input_ids"]
        attention_mask = context_tokenizer_outputs["attention_mask"]

        generation_result = model.forward(
            input_ids,
            attention_mask=attention_mask,
            metadata=[{"nl": test_input_str[0]}, {"nl": test_input_str[1]}],
        )

        self.assertEqual(len(generation_result), 2)
        self.assertEqual(
            all(["generated_program" in result for result in generation_result]), True
        )
