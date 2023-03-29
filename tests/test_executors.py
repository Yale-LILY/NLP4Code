import unittest

from execution.executors import MathExecutor

class TestExecutors(unittest.TestCase):
    def test_math_executor(self):
        executor = MathExecutor()

        test_program = "answer = 5"
        exec_match, exec_results = executor.exec_program(test_program, {"question": "some question", "answer": 5})
        
        self.assertEqual(exec_match, True)
        self.assertEqual(exec_results, {"answer": "5"})