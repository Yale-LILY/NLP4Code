import numpy as np
import pandas as pd
import io
from scipy import integrate
from scipy import misc
from scipy.ndimage import rotate
import numpy as np


def test(result, ans):
    res, (x1, y1) = result
    answer, (x_ans, y_ans) = ans
    assert np.allclose((x1, y1), (x_ans, y_ans))
    assert np.allclose(res, answer)
    return 1
