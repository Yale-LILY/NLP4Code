import tensorflow as tf


def test(result, ans=None):
    try:
        assert abs(result - ans) <= 0.02
        return 1
    except:
        return 0
