import numpy as np
import os
import random


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def seed_everything(seed=114514):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def frobenius_norm_sq(matrix):
    return np.linalg.norm(matrix, 'fro')**2


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


def residual_and_error_qr(matrix, indices):
    A, I = matrix, indices
    if not I:
        return A.copy(), frobenius_norm_sq(A)
    
    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()
    
    S = A[:, I]
    if S.shape[1] == 0:
        return A.copy(), frobenius_norm_sq(A)
    
    n, d = S.shape
    
    try:
        if n >= d:
            Q, R = np.linalg.qr(S, mode='reduced')
            S_pinv_A = np.linalg.solve(R, Q.T @ A)
            residual = A - S @ S_pinv_A
            error_val = frobenius_norm_sq(residual)
        else:
            S_pinv = np.linalg.pinv(S)
            residual = S @ S_pinv @ A
            error_val = frobenius_norm_sq(A - residual)
    except np.linalg.LinAlgError:
        print(f"Warning: QR decomposition failed for S with columns {I}. Assigning high error.")
        residual = A.copy()
        error_val = float('inf')
    
    return residual, error_val