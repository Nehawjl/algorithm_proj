import numpy as np
import os
import random


def rel_error(lhs, rhs):
    return np.max(np.abs(lhs - rhs) / (lhs + rhs))


def seed_everything(seed=114514):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def frobenius_norm_sq(matrix):
    return np.linalg.norm(matrix, 'fro')**2


def residual_error(matrix, indices):
    """
    Objective function: Calculates ||A - S S_pinv A||_F^2
    """
    A, I = matrix, indices
    if not I:
        return frobenius_norm_sq(A)
    
    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()

    S = A[:, I]
    if S.shape[1] == 0:
        return frobenius_norm_sq(A)

    try:
        S_pinv = np.linalg.pinv(S)
        error_val = frobenius_norm_sq(A - S @ S_pinv @ A)
        # S_pinv_A = np.linalg.lstsq(S, A, rcond=None)[0]
        # residual = A - S @ S_pinv_A
        # error_val = frobenius_norm_sq(residual)
    except np.linalg.LinAlgError:
        print(f"Warning: SVD did not converge for S with columns {I}. Assigning high error.")
        error_val = float('inf')    
    return error_val


def residual(matrix, indices):
    A, I = matrix, indices
    if not I:
        return A.copy()
    
    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()

    S = A[:, I]
    if S.shape[1] == 0:
        return A.copy()

    try:
        S_pinv = np.linalg.pinv(S)
        residual = A - S @ S_pinv @ A
        # S_pinv_A = np.linalg.lstsq(S, A, rcond=None)[0]
        # residual = A - S @ S_pinv_A
    except np.linalg.LinAlgError:
        print(f"Warning: SVD did not converge for S with columns {I}. Assigning high error.")
        residual = A.copy()
    return residual


def residual_and_error(matrix, indices):
    """
    Returns residual `A - S S_pinv A` and objective value
    """
    A, I = matrix, indices
    if not I:
        return A.copy(), frobenius_norm_sq(A)
    
    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()

    S = A[:, I]
    if S.shape[1] == 0:
        return A.copy(), frobenius_norm_sq(A)

    try:
        S_pinv = np.linalg.pinv(S)
        residual = A - S @ S_pinv @ A
        # S_pinv_A = np.linalg.lstsq(S, A, rcond=None)[0]
        # residual = A - S @ S_pinv_A
        error_val = frobenius_norm_sq(residual)
    except np.linalg.LinAlgError:
        print(f"Warning: SVD did not converge for S with columns {I}. Assigning high error.")
        residual = A.copy()
        error_val = float('inf')    
    return residual, error_val