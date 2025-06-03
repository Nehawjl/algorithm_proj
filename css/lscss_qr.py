import numpy as np
import math
from scipy.linalg import qr_delete, qr_insert

from .utility import frobenius_norm_sq


def error_from_qr(A_prime, Q, R, A_prime_norm_sq=None):
    # A_prime is the full matrix, Q & R are from A_prime[:, indices]
    # Error = ||A' - P_S A'||_F^2 = ||A'||_F^2 - ||Q^T A'||_F^2
    if Q is None or Q.shape[1] == 0:
        if A_prime_norm_sq is None:
            A_prime_norm_sq = frobenius_norm_sq(A_prime)
        return A_prime_norm_sq

    if A_prime_norm_sq is None:
        A_prime_norm_sq = frobenius_norm_sq(A_prime)

    # Q.T @ A_prime
    Q_transpose_A_prime = Q.T @ A_prime
    # ||Q^T A'||_F^2
    norm_sq_Q_T_A_prime = frobenius_norm_sq(Q_transpose_A_prime)

    error_val = A_prime_norm_sq - norm_sq_Q_T_A_prime
    return max(error_val, 0) # Ensure non-negative


def residual_from_qr(A_prime, Q, R):
    if Q is None or Q.shape[1] == 0:
        return A_prime.copy()
    # Residual = A_prime - Q @ (Q.T @ A_prime)
    # This is slightly different from A_prime - S @ S_pinv @ A_prime if S is rank deficient
    # but for full rank S, S_pinv_A = solve(R, Q.T @ A_prime)
    # residual = A_prime - Q @ Q.T @ A_prime
    # For column selection, Q Q^T A projects A onto the column space of S
    # So A - Q Q^T A is the residual.
    return A_prime - Q @ (Q.T @ A_prime)


def get_initial_qr(matrix, indices):
    if not indices:
        return None, None, frobenius_norm_sq(matrix)

    S = matrix[:, indices]
    if S.shape[1] == 0:
        return None, None, frobenius_norm_sq(matrix)

    try:
        Q, R = np.linalg.qr(S, mode='reduced')
        A_prime_norm_sq = frobenius_norm_sq(matrix)
        error = error_from_qr(matrix, Q, R, A_prime_norm_sq)
        return Q, R, error
    except np.linalg.LinAlgError:
        print(f"Warning: Initial QR decomposition failed for S with columns {indices}. Assigning high error.")
        return None, None, float('inf')


def residual_and_error_qr_original(matrix, indices):
    A, I = matrix, indices
    if not I:
        return A.copy(), frobenius_norm_sq(A)

    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()

    S = A[:, I]
    if S.shape[1] == 0:
        return A.copy(), frobenius_norm_sq(A)

    n_rows_S, n_cols_S = S.shape

    try:
        if n_rows_S >= n_cols_S:
            # Use QR decomposition for pseudo-inverse calculation if S is tall or square
            # Simpler using projection: A - Q_s Q_s^T A
            # This is equivalent if S has full column rank.
            Q_s, R_s = np.linalg.qr(S, mode='reduced')
            P_S_A = Q_s @ (Q_s.T @ A) # Project A onto column space of S
            residual_matrix = A - P_S_A
            error_val = frobenius_norm_sq(residual_matrix)
        else:
            # S is wide (more columns than rows), use SVD-based pinv for S S^+ A
            S_pinv = np.linalg.pinv(S) # S^+
            P_S_A = S @ S_pinv @ A
            residual_matrix = A - P_S_A # This is A_perp_to_range_S
            error_val = frobenius_norm_sq(residual_matrix)

    except np.linalg.LinAlgError:
        print(f"Warning: QR decomposition or pinv failed for S with columns {I}. Assigning high error.")
        residual_matrix = A.copy() # Or np.full_like(A, np.nan) or some other indicator
        error_val = float('inf')

    return residual_matrix, error_val


def get_col_sampling_probs(E_residual):
    col_norms_sq = np.sum(np.square(E_residual), axis=0)
    sum_col_norms_sq = np.sum(col_norms_sq)
    if sum_col_norms_sq == 0:
        return np.array([])
    sampling_probs = col_norms_sq / sum_col_norms_sq
    return sampling_probs


def ls_step_qr(A_prime, k, current_S_indices, Q_current, R_current, current_objective):
    n_rows, d_cols = A_prime.shape
    current_S_indices = list(current_S_indices)

    # ||e_j||^2 = ||a'_j - Q Q^T a'_j||^2
    if Q_current is not None:
        E_residual = residual_from_qr(A_prime, Q_current, R_current)
    else:
        E_residual = A_prime.copy()

    sampling_probs = get_col_sampling_probs(E_residual)
    if sampling_probs.size == 0:
        return current_S_indices, Q_current, R_current, current_objective

    num_candidates = 10 * k
    candidate_indices = np.random.choice(d_cols, size=num_candidates, p=sampling_probs, replace=True)
    p_swap_in = np.random.choice(candidate_indices)

    best_indices = current_S_indices
    best_objective = current_objective
    best_Q, best_R = Q_current, R_current

    A_prime_norm_sq = frobenius_norm_sq(A_prime) # Precompute for error calculation
    for q_idx, q_swap_out_original_idx in enumerate(current_S_indices):

        if p_swap_in == q_swap_out_original_idx: # Swapping a column with itself
            continue

        # Perform QR downdate (remove q_swap_out_original_idx)
        if Q_current is None or R_current is None or Q_current.shape[1] == 0: # Should not happen if current_S_indices is not empty
            # This case means Q_current/R_current is for an empty set, which is problematic for delete
            # Fallback to full QR:
            temp_indices = current_S_indices.copy()
            temp_indices.pop(q_idx) # Remove by index in list
            if not temp_indices: # If list becomes empty
                 Q_after_delete, R_after_delete = None, None
            else:
                S_temp = A_prime[:, temp_indices]
                if S_temp.shape[1] > 0:
                    Q_after_delete, R_after_delete = np.linalg.qr(S_temp, mode='reduced')
                else: # Should be caught by "if not temp_indices"
                    Q_after_delete, R_after_delete = None, None
        else:
            try:
                Q_after_delete, R_after_delete = qr_delete(Q_current, R_current, q_idx, which='col')
                # print(f"after delete, Q_after_delete has shape: {Q_after_delete.shape}")
            except Exception as e: # Catch potential errors in qr_delete, e.g. if R becomes singular
                print(f"qr_delete failed: {e}. Falling back for column {q_swap_out_original_idx} at index {q_idx}")
                # Fallback: recompute QR for S without that column
                temp_indices_for_delete = current_S_indices.copy()
                # The column to delete from S is at index q_idx
                temp_indices_for_delete.pop(q_idx)
                if not temp_indices_for_delete:
                    Q_after_delete, R_after_delete = None, None
                else:
                    S_temp = A_prime[:, temp_indices_for_delete]
                    if S_temp.shape[1] > 0:
                         Q_after_delete, R_after_delete = np.linalg.qr(S_temp, mode='reduced')
                    else: # Should be caught by "if not temp_indices_for_delete"
                         Q_after_delete, R_after_delete = None, None


        # Perform QR update (add p_swap_in)
        # The new column vector
        # col_to_add_vec = A_prime[:, [p_swap_in]] # Keep it as a 2D column vector
        col_to_add_vec = A_prime[:, p_swap_in] # Keep it as a 2D column vector

        # We want to insert it at the same position 'q_idx'
        if Q_after_delete is None or R_after_delete is None or Q_after_delete.shape[1] == 0: # If S_deleted was empty
            # This means we are just adding one column
            if col_to_add_vec.shape[1] > 0 : # Should always be true
                # Q_new, R_new = np.linalg.qr(col_to_add_vec, mode='reduced')
                Q_new, R_new = np.linalg.qr(col_to_add_vec.reshape(-1, 1), mode='reduced')
            else: # Should not happen
                Q_new, R_new = None, None
        else:
            try:
                # Insert the new column vector at logical position q_idx.
                # Note: qr_insert's k is the index in the *new* matrix where u will be.
                # If Q_after_delete has k-1 columns, inserting at q_idx (0 to k-1) makes sense.
                Q_new, R_new = qr_insert(Q_after_delete, R_after_delete, col_to_add_vec, q_idx, which='col')
            except Exception as e:
                print(f"qr_insert failed: {e}. Falling back for column {p_swap_in} at index {q_idx}")
                print(f"Q_after_delete has shape MxM {Q_after_delete.shape}, and col_to_add_vec has shape {col_to_add_vec.shape}")
                # Fallback: recompute QR
                temp_indices_for_insert = current_S_indices.copy()
                temp_indices_for_insert[q_idx] = p_swap_in # Replace
                S_new_temp = A_prime[:, temp_indices_for_insert]
                if S_new_temp.shape[1] > 0:
                    Q_new, R_new = np.linalg.qr(S_new_temp, mode='reduced')
                else:
                    Q_new, R_new = None, None


        # Calculate new objective function value
        f_new = error_from_qr(A_prime, Q_new, R_new, A_prime_norm_sq)

        if f_new < best_objective:
            best_objective = f_new
            new_indices_list = current_S_indices.copy()
            new_indices_list[q_idx] = p_swap_in
            best_indices = new_indices_list
            best_Q, best_R = Q_new, R_new

    return best_indices, best_Q, best_R, best_objective


def lscss_algorithm_qr(A, k, T=None):
    n_rows, d_cols = A.shape

    if k == 0:
        return []
    if k >= d_cols:
        return list(range(d_cols))
    if T is None:
        T = max(1, int(k**2 * np.log(k + 1)))

    B = A.copy() # Original A
    A_current_for_greedy = A.copy() # This A might be modified by adding D

    # Phase 1: Add D matrix
    for t in range(1, 3): # Runs for t=1 and t=2
        I_indices = [] # Stores original column indices
        # For the greedy selection, we manage Q, R incrementally for A_current_for_greedy[:, I_indices]
        Q_greedy, R_greedy = None, None # QR of A_current_for_greedy[:, I_indices]

        # A_current_for_greedy_norm_sq = frobenius_norm_sq(A_current_for_greedy) # For error calculation

        for iter_k in range(k): # Greedily select k columns
            # Calculate residual based on Q_greedy, R_greedy for A_current_for_greedy
            if Q_greedy is not None:
                E_residual = residual_from_qr(A_current_for_greedy, Q_greedy, R_greedy)
            else:
                E_residual = A_current_for_greedy.copy()

            probs = get_col_sampling_probs(E_residual)
            if probs.size == 0: # No columns left to sample or residual is zero
                break

            available_mask = np.ones(d_cols, dtype=bool)
            if I_indices: # If I_indices is not empty
                available_mask[I_indices] = False

            # Filter out already selected indices
            # Ensure there are available columns to sample from
            if not np.any(available_mask): break # No more columns to select
            
            active_probs = probs[available_mask]
            sum_active_probs = np.sum(active_probs)
            if sum_active_probs == 0 : # If all remaining columns have zero probability
                 # Fallback: uniform probability over available unselected columns
                num_available = np.sum(available_mask)
                if num_available == 0: break
                active_probs = np.ones(num_available) / num_available
            else:
                active_probs = active_probs / sum_active_probs


            available_indices = np.where(available_mask)[0]
            sampled_idx_in_available = np.random.choice(len(available_indices), p=active_probs)
            sampled_original_idx = available_indices[sampled_idx_in_available]

            # Update Q_greedy, R_greedy by adding the new column
            # new_col_vec = A_current_for_greedy[:, [sampled_original_idx]] # Keep as 2D column vector
            new_col_vec = A_current_for_greedy[:, sampled_original_idx] # Keep as 2D column vector

            if Q_greedy is None or Q_greedy.shape[1] == 0: # First column
                # Q_greedy, R_greedy = np.linalg.qr(new_col_vec, mode='reduced')
                Q_greedy, R_greedy = np.linalg.qr(new_col_vec.reshape(-1, 1), mode='reduced')
            else:
                try:
                    # Insert at the end of the current set of columns
                    # The index for insertion is Q_greedy.shape[1] (0-indexed, so if 1 col, insert at index 1)
                    Q_greedy, R_greedy = qr_insert(Q_greedy, R_greedy, new_col_vec, Q_greedy.shape[1], which='col')
                except Exception as e:
                    print(f"Greedy qr_insert failed: {e}. Falling back for col {sampled_original_idx}")
                    # Fallback: full QR on new set
                    temp_greedy_indices = I_indices + [sampled_original_idx]
                    S_greedy_temp = A_current_for_greedy[:, temp_greedy_indices]
                    if S_greedy_temp.shape[1] > 0:
                         Q_greedy, R_greedy = np.linalg.qr(S_greedy_temp, mode='reduced')
                    else: # Should not happen
                         Q_greedy, R_greedy = None, None


            I_indices.append(sampled_original_idx)
            if len(I_indices) == k:
                break
        
        # After greedy selection for current A_current_for_greedy
        if t == 1:
            # Calculate residual_norm_sq for B (original A) using the selected I_indices
            # This uses the original full residual calculation
            _, residual_norm_sq_for_B = residual_and_error_qr_original(B, I_indices)
            residual_norm = np.sqrt(max(residual_norm_sq_for_B,0)) # Ensure non-negative

            try:
                factorial_term = math.factorial(k + 1) # Can overflow for large k
                denominator = np.sqrt(52 * min(n_rows, d_cols) * factorial_term)
                alpha = residual_norm / denominator if denominator > 0 else 1e-10 # Avoid division by zero
            except (OverflowError, ValueError, TypeError): # Handle large k for factorial
                log_factorial = sum(np.log(i) for i in range(1, k + 2)) # log((k+1)!)
                log_denominator = 0.5 * (np.log(52) + np.log(min(n_rows, d_cols)) + log_factorial)
                alpha = residual_norm * np.exp(-log_denominator) if residual_norm > 0 else 1e-10


            # D is diagonal matrix. Adding it to A and B
            # Ensure D is compatible with A for addition
            D_diag = np.zeros(min(n_rows, d_cols)) 
            D_diag[:] = alpha # Fill with alpha
            
            # A_current_for_greedy will be A + D for the next iteration's greedy selection
            # A_prime for local search will be B + D
            # Creating D explicitly might be memory intensive if n_rows, d_cols are huge.
            # However, A = A + D implies A must be modified.
            # If A is sparse, this would densify it. Assuming A is dense.
            
            # This modification of A_current_for_greedy means its QR needs re-evaluation in the next loop
            # or careful consideration. The original paper implies A_prime is B+D.
            # And A is A_orig+D. The greedy selection in second pass uses A_orig+D.
            
            # Let's define A_prime for the local search step first
            A_prime = B.copy() # Start from original B
            min_dim_prime = min(A_prime.shape[0], A_prime.shape[1])
            A_prime[np.arange(min_dim_prime), np.arange(min_dim_prime)] += alpha

            # For the next greedy pass (t=2), A_current_for_greedy should be A_orig + D
            A_current_for_greedy = A.copy() # Start from original A
            min_dim_A_current = min(A_current_for_greedy.shape[0], A_current_for_greedy.shape[1])
            A_current_for_greedy[np.arange(min_dim_A_current), np.arange(min_dim_A_current)] += alpha

    # S_indices are from the *second* pass of greedy selection (using A_orig + D)
    S_indices = I_indices # These are the indices from t=2 run

    # Initial QR for S_indices on A_prime (which is B+D)
    # Note: S_indices were chosen based on A_current_for_greedy (A_orig+D),
    # but local search operates on A_prime (B+D).
    # If A=B (i.e. no pre-processing on A differing from B), then A_current_for_greedy = A_prime.
    Q_ls, R_ls, current_objective_ls = get_initial_qr(A_prime, S_indices)
    if Q_ls is None and k > 0 : # Failed initial QR for local search and k > 0
        print("Warning: Initial QR for local search failed. Returning S_indices from greedy.")
        return S_indices


    # Phase 2: Local Search using A_prime = B + D
    for i in range(T):
        if not S_indices and k > 0 : # If S_indices became empty somehow but k > 0
             print(f"Warning: S_indices is empty at iteration {i} of local search.")
             # Potentially re-initialize S_indices or break
             # For now, break, or re-initialize with a random set of k indices
             # Fallback to a random set of k indices from d_cols
             if d_cols >= k:
                 S_indices = list(np.random.choice(d_cols, k, replace=False))
                 Q_ls, R_ls, current_objective_ls = get_initial_qr(A_prime, S_indices)
                 if Q_ls is None: # If even random selection fails QR
                     print("Fallback random selection also failed QR. Returning previous S_indices.")
                     break # Or return the last valid S_indices
             else: # Should not happen due to initial checks
                 break


        S_indices, Q_ls, R_ls, current_objective_ls = ls_step_qr(
            A_prime, k, S_indices, Q_ls, R_ls, current_objective_ls
        )
        # print(f"LS iter {i}, objective: {current_objective_ls}, indices: {S_indices}")


    return S_indices