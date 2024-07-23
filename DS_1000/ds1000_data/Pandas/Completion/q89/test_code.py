import pandas as pd
import numpy as np


def test(result, ans=None):
    try:
        assert (result == ans).all()
        return 1
    except:
        return 0
