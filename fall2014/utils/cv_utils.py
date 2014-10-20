__author__ = 'nikita_kartashov'
import numpy as np


def make_kernel(width, height):
    return np.ones((width, height), np.uint8)


def make_square_kernel(n):
    return make_kernel(n, n)
