import numpy as np
import os
import random


def seed_everything(seed=114514):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def frobenius_norm_sq(matrix):
    return np.linalg.norm(matrix, 'fro')**2