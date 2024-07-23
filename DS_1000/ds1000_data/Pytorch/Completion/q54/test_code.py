import argparse
import os
import pickle
import numpy as np
import torch


def test(result, ans=None):
    
    
    try:
        assert len(ans) == len(result)
        for i in range(len(ans)):
            torch.testing.assert_close(result[i], ans[i], check_dtype=False)
        return 1
    except:
        return 0





































