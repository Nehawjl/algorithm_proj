from itertools import combinations
import numpy as np

from .utility import residual_and_error_qr


def brute_force_css(A, k):
    _, d = A.shape
    
    if k == 0: return []
    if k >= d: return list(range(d))
    
    best_indices = None
    best_objective = float('inf')

    for col_indices in combinations(range(d), k):

        col_indices_list = list(col_indices)
        objective = residual_and_error_qr(A, col_indices_list)[1]

        if objective < best_objective:
            best_objective = objective
            best_indices = col_indices_list
    
    return best_indices