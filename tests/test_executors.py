import unittest

from os import path, sys

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from execution.executors import MathExecutor
from tests.consts import TEST_EXECUTORS


class TestExecutors(unittest.TestCase):
    def test_executors(self):
        for executor_cls, test_program, test_example in TEST_EXECUTORS:
            print(f"\n======== testing {executor_cls.__name__} ========")
            executor = executor_cls()

            print(test_program)
            print(test_example)

            try:
                exec_match, exec_results = executor.exec_program(
                    test_program, test_example
                )
                self.assertIsInstance(exec_match, int)
                print(exec_results)
            # TODO: use real DB connections
            except:
                self.assertIsInstance(exec_match, int)
                print(exec_results)

    # custom tests for specific executors

    def test_math_executor(self):
        executor = MathExecutor()

        test_program = "answer = 5"
        exec_match, exec_results = executor.exec_program(
            test_program, {"question": "some question", "answer": 5}
        )

        self.assertEqual(exec_match, True)
        self.assertEqual(exec_results, {"answer": "5"})
