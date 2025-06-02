import numpy as np
from utility import frobenius_norm_sq


def residual_error(A_orig, S_indices, A_for_S_cols=None):
    """
    Calculates ||A_orig - S S_dagger A_orig||_F^2
    S_indices: indices of columns selected.
    A_for_S_cols: The matrix from which columns for S are taken (e.g. A_orig or A_prime). If None, assumes A_orig.
    """
    if not S_indices: # No columns selected
        return frobenius_norm_sq(A_orig)
    
    A_select_from = A_for_S_cols if A_for_S_cols is not None else A_orig
    
    # Ensure S_indices is a list or 1D array for proper indexing
    if not isinstance(S_indices, (list, np.ndarray)) or (isinstance(S_indices, np.ndarray) and S_indices.ndim > 1):
        # This case can happen if S_indices becomes a 2D array accidentally
        # For safety, flatten or take the first row/column if it's a single list of indices
        S_indices = np.array(S_indices).flatten().tolist()


    S = A_select_from[:, S_indices]
    
    if S.shape[1] == 0: # No columns selected
        return frobenius_norm_sq(A_orig)

    try:
        S_dagger = np.linalg.pinv(S)
        projection_S_A = S @ S_dagger @ A_orig
        error_val = frobenius_norm_sq(A_orig - projection_S_A)
    except np.linalg.LinAlgError:
        print(f"Warning: SVD did not converge for S with columns {S_indices}. Assigning high error.")
        error_val = float('inf') # Or a very large number
        # Potentially, S has linearly dependent columns or is otherwise problematic
        # This can happen if k is too large relative to rank or due to numerical issues.
    return error_val