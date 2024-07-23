import numpy as np
import pandas as pd
import torch


def test(result, ans):
    assert type(result) == torch.Tensor
    torch.testing.assert_close(result, ans, check_dtype=False)
    return 1
