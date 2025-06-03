import numpy as np
import random
from utility import frobenius_norm_sq, residual_error


def greedy_css(A_orig, k):
    """
    A common greedy approach: iteratively add the column that maximally reduces the current error.
    This is computationally intensive due to repeated pinv.
    """
    n, d = A_orig.shape
    if k == 0: return frobenius_norm_sq(A_orig)
    if k >= d: 
        all_indices = list(range(d))
        return all_indices

    selected_indices = []
    remaining_indices = list(range(d))
    
    current_A_for_S = A_orig # Columns are from A_orig

    for _ in range(k):
        if not remaining_indices: break # No more columns to select

        best_new_idx = -1
        min_error_for_this_step = float('inf')

        for candidate_idx in remaining_indices:
            temp_selection = selected_indices + [candidate_idx]
            error = residual_error(A_orig, temp_selection)
            
            if error < min_error_for_this_step:
                min_error_for_this_step = error
                best_new_idx = candidate_idx
        
        if best_new_idx != -1:
            selected_indices.append(best_new_idx)
            remaining_indices.remove(best_new_idx)
        else: # No column improved error (or remaining_indices was empty)
            break 
            
    # final_error = residual_error(A_orig, selected_indices, A_for_S_cols=current_A_for_S)
    return selected_indices


def greedy_recursive_css(A_orig, k):
    """
    Implements GreedyFS using recursive updates for G = E^T E.
    Selects feature l = argmax_i ( ||G_{:i}||^2 / G_{ii} )
    """
    n_samples, d_features = A_orig.shape
    if k == 0: return [], np.linalg.norm(A_orig, 'fro')**2
    if k >= d_features:
        all_indices = list(range(d_features))
        # For full selection, error is 0 if A can be perfectly reconstructed by itself
        # P_S A = A (A.T A)^-1 A.T A = A if A.T A is invertible.
        # If k=d, A_S = A. P_S A = A (A^T A)^† A^T A.
        # If A has full column rank, (A^T A)^† A^T A = I, so P_S A = A. Error = 0.
        # Otherwise, it's the projection onto its own column space, so error is 0.
        return all_indices, 0.0

    selected_indices = []
    # remaining_indices 是一个 boolean 掩码，True 表示可选
    available_indices_mask = np.ones(d_features, dtype=bool)

    # Initial E = A, so G_current = A^T A
    G_current = A_orig.T @ A_orig
    
    # Small epsilon for numerical stability in division
    eps = 1e-12

    for iter_num in range(k):
        best_new_idx = -1
        max_score_for_this_step = -float('inf')

        # Iterate over indices that are still available
        for candidate_idx in np.where(available_indices_mask)[0]:
            G_ii = G_current[candidate_idx, candidate_idx]
            if G_ii < eps: # Avoid division by zero or near-zero (col likely zero or dependent)
                score = -float('inf') # effectively skip this candidate
            else:
                G_col_i = G_current[:, candidate_idx]
                score = np.linalg.norm(G_col_i)**2 / G_ii
            
            if score > max_score_for_this_step:
                max_score_for_this_step = score
                best_new_idx = candidate_idx
        
        if best_new_idx != -1:
            selected_indices.append(best_new_idx)
            available_indices_mask[best_new_idx] = False # Mark as selected

            # Update G for the next iteration
            # G_new = G_old - (G_old_{:l} G_old_{:l}^T) / G_old_{ll}
            G_col_l = G_current[:, best_new_idx].reshape(-1, 1) # Ensure it's a column vector
            G_ll = G_current[best_new_idx, best_new_idx]
            
            if G_ll < eps: # Should not happen if selected, but for safety
                # This means the selected column has no energy in the residual space,
                # which implies further selections won't reduce error.
                print(f"Warning: G_ll near zero for selected feature {best_new_idx} in iter {iter_num}. Stopping.")
                break
            
            G_current = G_current - (G_col_l @ G_col_l.T) / G_ll
        else: # No suitable feature found (e.g., all remaining G_ii are zero)
            break
            
    # Final error calculation F(S) = ||A - P^(S)A||_F^2
    # This G_current is E^T E where E = A - P^(S_final) A.
    # F(S) = trace(E^T E) = trace(G_current)
    # However, the F(S) = F(S_prev) - ||E_tilde_l||^2_F relation applies to the *original* F(S).
    # The G update is for the selection criterion.
    # To get the final error value, it's safest to recompute with calculate_objective_value.
    # final_error = calculate_objective_value(A_orig, selected_indices)
    
    return selected_indices


def partition_greedy_css(A_orig, k_to_select, num_partitions_c):
    """
    Implements Partition-based GreedyFS using Algorithm 1 from the paper
    and recursive updates for f_i and g_i from Theorem 5.
    """
    n_samples, d_features = A_orig.shape
    eps = 1e-12 # Small constant for numerical stability

    if k_to_select == 0: return [], np.linalg.norm(A_orig, 'fro')**2
    if k_to_select >= d_features:
        return list(range(d_features)), 0.0 # Error is 0 if all features are selected

    selected_indices = []
    available_mask = np.ones(d_features, dtype=bool)

    # --- Step 1: Initialization from Algorithm 1 ---
    # Generate a random partitioning P
    if num_partitions_c <= 0 or num_partitions_c > d_features:
        num_partitions_c = max(1, min(d_features, int(d_features / 10))) # Heuristic for c
        print(f"Adjusted num_partitions_c to {num_partitions_c}")

    all_feature_indices = list(range(d_features))
    random.shuffle(all_feature_indices)
    partitions = [list(arr) for arr in np.array_split(all_feature_indices, num_partitions_c)]

    # Calculate B: B_j = sum_{r in P_j} A_{:r}
    B = np.zeros((n_samples, num_partitions_c))
    for j, p_j in enumerate(partitions):
        if p_j: # If partition is not empty
            B[:, j] = np.sum(A_orig[:, p_j], axis=1)
            # Paper mentions normalization for B later, but initial def is sum.
            # "The use of B suimhn the size of the corresponding group." - seems like a typo, might mean "normalized by"
            # For now, let's stick to sum as in B_j = sum A_r

    # --- Step 2: Initialize f_i^(0) and g_i^(0) ---
    f_current = np.zeros(d_features)
    g_current = np.zeros(d_features)
    
    A_T_A_diag = np.einsum('ij,ij->j', A_orig, A_orig) # A_orig[:,i].T @ A_orig[:,i]

    B_T_A = B.T @ A_orig # c x d_features

    for i in range(d_features):
        # f_i^(0) = ||B^T A_{:i}||^2
        f_current[i] = np.linalg.norm(B_T_A[:, i])**2
        # g_i^(0) = A_{:i}^T A_{:i}
        g_current[i] = A_T_A_diag[i]

    all_omegas = [] # List to store omega^(r) vectors
    all_vs = []     # List to store v^(r) vectors

    # Precompute A^T A and A^T B for updates (as per paper's complexity discussion)
    A_T_A_full = A_orig.T @ A_orig # d_features x d_features
    A_T_B_full = A_orig.T @ B      # d_features x num_partitions_c

    # --- Step 3: Repeat t = 1 to k_to_select ---
    for t_iter in range(k_to_select):
        # a) Select l = argmax_i (f_i / g_i)
        scores = np.full(d_features, -np.inf)
        valid_indices_for_selection = np.where(available_mask & (g_current > eps))[0]
        
        if not valid_indices_for_selection.size:
            print(f"No more valid features to select at iteration {t_iter+1}.")
            break
            
        scores[valid_indices_for_selection] = f_current[valid_indices_for_selection] / g_current[valid_indices_for_selection]
        
        l_selected_idx = np.argmax(scores)

        if scores[l_selected_idx] == -np.inf : # No selectable feature
             print(f"No suitable feature found to select at iteration {t_iter+1}.")
             break

        selected_indices.append(l_selected_idx)
        available_mask[l_selected_idx] = False

        # b) Calculate delta^(t)
        # delta_vector_l_t = A^T A_{:l} - sum_{r=1}^{t-1} omega_l^{(r)} omega^{(r)}
        delta_vec_t = A_T_A_full[:, l_selected_idx].copy() # This is A^T A_{:l}
        for r_idx in range(len(all_omegas)):
            omega_r = all_omegas[r_idx]
            omega_l_r = omega_r[l_selected_idx] # scalar
            delta_vec_t -= omega_l_r * omega_r
        
        # c) Calculate gamma^(t)
        # gamma_vector_l_t = B^T A_{:l} - sum_{r=1}^{t-1} omega_l^{(r)} v^{(r)}
        gamma_vec_t = B_T_A[:, l_selected_idx].copy() # This is B^T A_{:l}
        for r_idx in range(len(all_omegas)): # same loop as for delta
            omega_r = all_omegas[r_idx]
            v_r = all_vs[r_idx]
            omega_l_r = omega_r[l_selected_idx] # scalar
            gamma_vec_t -= omega_l_r * v_r

        # d) Calculate omega^(t) and v^(t)
        delta_scalar_l_t = delta_vec_t[l_selected_idx]

        if delta_scalar_l_t < eps:
            print(f"Warning: delta_scalar_l_t is near zero for feature {l_selected_idx} at iter {t_iter+1}. Stopping.")
            break
        
        sqrt_delta_l_t = np.sqrt(delta_scalar_l_t)
        omega_t_new = delta_vec_t / sqrt_delta_l_t
        v_t_new = gamma_vec_t / sqrt_delta_l_t
        
        all_omegas.append(omega_t_new)
        all_vs.append(v_t_new)

        # e) Update f_i's, g_i's (Theorem 5) for *next* iteration's selection
        # g_next_i = g_current_i - (omega_t_new[i])^2
        g_current = g_current - (omega_t_new**2) # Element-wise square

        # For f_next_i update:
        # f_next_i = f_current_i - 2*omega_t_new[i]*( A^T B v_t_new - sum_{r=1}^{t-1} (v^(r)^T v_t_new)omega^(r) )[i]
        #                     + ||v_t_new||^2 * (omega_t_new[i])^2
        
        term_in_parentheses = A_T_B_full @ v_t_new # (d_features x c) @ (c x 1) -> (d_features x 1)
        
        # The sum is over r=1 to t-1 (i.e., all omegas/vs *before* the current omega_t_new/v_t_new)
        # all_omegas[:-1] and all_vs[:-1] would be appropriate if omega_t_new is already appended.
        # Since we just computed omega_t_new and v_t_new, the sum is over all_omegas and all_vs *before* appending them.
        # So, the sum should be over all_omegas[0]...all_omegas[len-2] if len refers to after appending.
        # Or, more simply, sum over all_omegas as they are *before* appending the current one.
        # The paper uses sum_{r=1}^{t-2} for G*omega in Theorem 4 proof, relative to omega being omega^(t-1).
        # If omega_t_new is omega^(t_iter), sum over r=0 to t_iter-1 (previous omegas/vs)
        
        # Let's use the structure from Theorem 5 explicitly:
        # G_omega_term = A^T B v - sum_{r=1}^{t-1} (v^(r)^T v) omega^(r)
        # Here, v is v_t_new, and sum is over previous v^(r) and omega^(r)
        # `all_omegas` and `all_vs` *before* appending current omega_t_new and v_t_new
        # are effectively omega^(1) ... omega^(t-1) if t_iter is current time t.
        
        sum_val = np.zeros(d_features)
        if t_iter > 0: # if there are previous omegas/vs
            for r_idx in range(len(all_omegas) -1): # Exclude the just added omega_t_new/v_t_new
                v_r = all_vs[r_idx]
                omega_r = all_omegas[r_idx]
                v_r_T_v_t_new = v_r.T @ v_t_new # scalar
                sum_val += v_r_T_v_t_new * omega_r
        
        term_in_parentheses -= sum_val
        
        # Hadamard product `o` is element-wise multiplication `*` in numpy
        f_current = f_current - 2 * (omega_t_new * term_in_parentheses) + np.linalg.norm(v_t_new)**2 * (omega_t_new**2)

    # final_error = calculate_objective_value(A_orig, selected_indices)
    return selected_indices