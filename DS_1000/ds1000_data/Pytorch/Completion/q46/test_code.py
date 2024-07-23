import argparse
import os
import pickle
import numpy as np
import torch


def test(result, ans=None):
    
    
    try:
        assert result.type() == "torch.LongTensor"
        torch.testing.assert_close(result, ans)
        return 1
    except:
        return 0





































