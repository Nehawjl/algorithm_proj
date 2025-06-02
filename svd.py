import numpy as np
from utility import frobenius_norm_sq


def svd_error(A, k):
    """
    Computes ||A - A_k||_F^2 using SVD
    A_k is the best rank-k approximation.
    """
    if k == 0:
        return frobenius_norm_sq(A)

    if k >= min(A.shape): # If k is full rank or more
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        A_k_approx = U @ np.diag(s) @ Vt
        return frobenius_norm_sq(A - A_k_approx)

    try:
        _, s_all, _ = np.linalg.svd(A, full_matrices=False)
        if k >= len(s_all): # if k is equal or larger than rank
            return 0.0 
        # For this error, we only need the singular values not chosen
        error_A_k = np.sum(s_all[k:]**2)
        return error_A_k
    except Exception as e:
        print(f"SVD for A_k failed: {e}. Falling back to full SVD error calculation.")
        # Fallback if randomized_svd has issues or for exactness
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        s_full_rank = np.zeros(min(A.shape))
        s_full_rank[:len(s)] = s
        A_k_approx = U[:, :k] @ np.diag(s_full_rank[:k]) @ Vt[:k, :]
        return frobenius_norm_sq(A - A_k_approx)
