from css.random_css import random_css
from css.greedy_css import greedy_css, greedy_recursive_css, partition_greedy_css
from css.lscss import lscss_algorithm
from css.bf_css import brute_force_css
from css.utility import residual_error


class CSSProblemSolver:
    def __init__(self):
        self.solvers = {
            # brute force
            'bf': brute_force_css,

            # random
            'random': random_css,

            # greedy methods
            'greedy': greedy_css,
            'greedy_rec': greedy_recursive_css,
            'greedy_par': partition_greedy_css,

            # lscss
            'lscss': lscss_algorithm,
        }
        
    def solve(self, method_name, *args, **kwargs):
        if method_name not in self.solvers:
            raise ValueError(f"Unknown method: {method_name}. Methods should be within: {list(self.solvers.keys())}")
        return self.solvers[method_name](*args, **kwargs)
    
    def get_available_methods(self):
        return list(self.solvers.keys())

    def get_objective(self, matrix, indices):
        return residual_error(matrix, indices)
