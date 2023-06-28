import tensorflow as tf


def test(result, ans=None):
    try:
        assert result == ans
        return 1
    except:
        return 0
