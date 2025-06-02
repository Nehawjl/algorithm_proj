import numpy as np
import random


def random_css(A, k) -> list[int]:
    """return selected indices"""
    _, d = A.shape
    I = list(range(d)) if k >= d else random.sample(range(d), k)
    return I