import numpy as np

from bayer.scripts.darkflat import _average


def test_average_mean():
    ones = np.ones((2, 2), dtype=np.float32)
    stack = np.stack([ones, ones*2, ones*6], axis=0)
    out = _average(stack, 'mean')
    np.testing.assert_allclose(ones * (1 + 2 + 6) / 3, out)


def test_average_median():
    ones = np.ones((2, 2), dtype=np.float32)
    stack = np.stack([ones, ones*2, ones*6], axis=0)
    out = _average(stack, 'median')
    np.testing.assert_allclose(ones*2, out)


def test_average_sigma3():
    zeros = np.zeros((2, 2), dtype=np.float32)
    ones = np.ones((2, 2), dtype=np.float32)
    stack = np.stack([zeros, zeros, zeros, ones*100, -ones*100], axis=0)
    out = _average(stack, 'sigma3')
    np.testing.assert_allclose(zeros, out)
