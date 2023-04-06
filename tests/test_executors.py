import unittest

from os import path, sys

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from execution.executors import MathExecutor


class TestExecutors(unittest.TestCase):
    def test_math_executor(self):
        executor = MathExecutor()

        test_program = "answer = 5"
        exec_match, exec_results = executor.exec_program(
            test_program, {"question": "some question", "answer": 5}
        )

        self.assertEqual(exec_match, True)
        self.assertEqual(exec_results, {"answer": "5"})
