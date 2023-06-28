import argparse
import os
import pickle
import numpy as np
import torch


def test(result, ans=None):
    
    
    try:
        torch.testing.assert_close(result, ans, check_dtype=False)
        return 1
    except:
        return 0


























