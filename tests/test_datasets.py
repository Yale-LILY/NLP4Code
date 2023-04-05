import unittest

from os import path, sys

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from execution.executors import MathExecutor


class TestDatasets(unittest.TestCase):
    def test_gsmath(self):
        # TODO: this is dummy test
        self.assertTrue(True)
