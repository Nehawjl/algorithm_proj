import numpy as np
import random
import math
from utility import residual_error


def get_col_sampling_probs(E_matrix):
    """
    Calculates p_i = ||E_matrix_{:,i}||_2^2 / ||E_matrix||_F^2
    Returns a 1D array of probabilities.
    """
    if E_matrix.shape[1] == 0:
        return np.array([])

    col_norms_sq = np.sum(np.square(E_matrix), axis=0) # Sum squares down columns
    total_norm_sq = np.sum(col_norms_sq) # This is ||E_matrix||_F^2

    if total_norm_sq < 1e-20: # Avoid division by zero if E_matrix is (close to) zero
        # Uniform probability if matrix is zero
        return np.ones(E_matrix.shape[1]) / E_matrix.shape[1]
    
    probs = col_norms_sq / total_norm_sq
    # Ensure probabilities sum to 1 due to potential floating point inaccuracies
    probs = probs / np.sum(probs)
    return probs

# --- Algorithm 2: LS (Local Search Step) ---
def ls_step(A_prime, k, current_S_indices_in_A_prime):
    """
    Performs one step of the local search (Algorithm 2).
    A_prime: The (potentially perturbed) matrix A'.
    k: Target number of columns.
    current_S_indices_in_A_prime: List of column indices currently in S, w.r.t A_prime.
    Returns: Updated list of column indices for S (w.r.t A_prime).
    """
    n_prime, d_prime = A_prime.shape
    
    # Ensure current_S_indices are valid and k
    # This function expects current_S_indices to be valid and of length k.
    # LSCSS main algorithm should ensure this before calling.
    if len(current_S_indices_in_A_prime) != k:
        # print(f"Warning (LS_step): Input S_indices length {len(current_S_indices_in_A_prime)} != k {k}. Problems might occur.")
        # This path should ideally not be taken if called correctly.
        # If it happens, might need to re-select or pad, but that's fragile.
        # For now, assume it's k.
        pass


    # Calculate current objective value f(A', S)
    # S is A_prime[:, current_S_indices_in_A_prime]
    # Error is ||A' - S S_dagger A'||_F^2
    f_current = residual_error(A_prime, current_S_indices_in_A_prime)

    # 1. Compute residual matrix E = A' - S S_dagger A'
    S_matrix = A_prime[:, current_S_indices_in_A_prime]
    try:
        S_dagger = np.linalg.pinv(S_matrix)
        E_residual = A_prime - S_matrix @ S_dagger @ A_prime
    except np.linalg.LinAlgError:
        # print("Warning (LS_step): SVD for S_dagger failed. Using A_prime as E_residual (implies high error for S).")
        E_residual = A_prime.copy() # Full residual

    # 2. Sample a list C of 10k column indices from A' (indices 0 to d_prime-1)
    # Probabilities p_i = ||E_residual_{:,i}||_2^2 / ||E_residual||_F^2
    if d_prime == 0: # No columns to sample from
        return sorted(list(set(current_S_indices_in_A_prime)))

    sampling_probs = get_col_sampling_probs(E_residual)
    
    num_candidates_C = 10 * k
    if num_candidates_C == 0 and d_prime > 0 : # k=0 case
        num_candidates_C = d_prime # sample all if k=0, though k=0 should be handled earlier
    elif num_candidates_C == 0 and d_prime == 0:
        return sorted(list(set(current_S_indices_in_A_prime)))


    candidate_indices_C = np.random.choice(d_prime, size=num_candidates_C, p=sampling_probs, replace=True)

    # 3. Uniformly sample an index p_swap_in from C
    if not candidate_indices_C.size: # Should not happen if d_prime > 0
        return sorted(list(set(current_S_indices_in_A_prime)))
    p_swap_in = random.choice(candidate_indices_C)

    # 4. (current_S_indices_in_A_prime are I)

    # 5. Try to find q_swap_out in I to improve f
    best_q_swap_out = -1 # Sentinel
    min_f_after_swap = f_current
    best_potential_S_indices = list(current_S_indices_in_A_prime) # Start with current

    for q_swap_out in current_S_indices_in_A_prime:
        # Tentative new set of indices: (I \ {q_swap_out}) U {p_swap_in}
        temp_S_indices_set = set(current_S_indices_in_A_prime)
        temp_S_indices_set.remove(q_swap_out)
        temp_S_indices_set.add(p_swap_in)

        # Ensure the new set still has k distinct columns
        if len(temp_S_indices_set) == k:
            potential_S_indices_list = sorted(list(temp_S_indices_set))
            f_temp = residual_error(A_prime, potential_S_indices_list)

            if f_temp < min_f_after_swap:
                min_f_after_swap = f_temp
                # best_q_swap_out = q_swap_out # Not strictly needed, best_potential_S_indices is key
                best_potential_S_indices = potential_S_indices_list
    
    # 6 & 7. If improvement found, update S_indices
    # The condition `min_f_after_swap < f_current` is implicitly handled by how `best_potential_S_indices` is updated.
    # If no swap is better, `best_potential_S_indices` remains `current_S_indices_in_A_prime`.
    return sorted(list(set(best_potential_S_indices)))


# --- Algorithm 1: LSCSS ---
def lscss_algorithm(A, k, T=None):
    """
    Implements the LSCSS algorithm (Algorithm 1).
    A: The original n x d matrix.
    k: Number of columns to select.
    T: Number of iterations for the local search part.
    Returns: Selected indices
    """
    n, d = A.shape

    if k == 0:
        return []
    if k >= d: # Select all columns if k >= d
        return list(range(d))
    T = T if T else max(1, int(k**2 * np.log(k + 1)))

    B = A.copy() # Step 1: B = A (original)
    A_tmp = A.copy() # This variable will hold A then A_prime
    D = np.zeros_like(A) # Initialize D, might be updated
    A_prime = None # Will be B + D

    # This will store the indices for A_prime after initialization step
    S_indices_for_LS_init_in_A_prime = [] 

    # --- Initialization: Steps 1-12 ---
    # Loop for t = 1, 2 (Pass 1 for D calc, Pass 2 for S_init on A')
    for t_pass in range(1, 3):
        I = set() # For current pass (Alg1 uses I)
        
        # Initial E for this pass. If I is empty, E is A_tmp.
        # Otherwise E = A_current - A_I A_I_dagger A_current
        E = A_tmp.copy()

        # 3. For j = 1 to k (sample k times to build/refine I)
        for _ in range(k): # k sampling attempts
            if A_tmp.shape[1] == 0: break # No columns left
            
            sampling_probs = get_col_sampling_probs(E)
            if not sampling_probs.size: # E is empty or all zero columns
                 # Fallback: if E is zero, sample uniformly from available columns in A_current
                available_cols_in_A_current = [c_idx for c_idx in range(A_tmp.shape[1]) if c_idx not in I]
                if not available_cols_in_A_current: break # No more unique columns to pick
                sampled_idx = random.choice(available_cols_in_A_current)
            else:
                sampled_idx = np.random.choice(A_tmp.shape[1], p=sampling_probs)
            
            I.add(sampled_idx) # Add to set (handles distinctness)
            
            # 5. Update E = A_current - A_I A_I_dagger A_current
            # Ensure current_I_indices is not empty before forming S_temp
            if I:
                # Indices must be list for slicing
                S_temp_indices_list = sorted(list(I)) 
                S_temp = A_tmp[:, S_temp_indices_list]
                try:
                    S_temp_dagger = np.linalg.pinv(S_temp)
                    E = A_tmp - S_temp @ S_temp_dagger @ A_tmp
                except np.linalg.LinAlgError:
                    # print(f"Warning (LSCSS Init t={t_pass}): SVD for S_temp_dagger failed. Using A_current as E.")
                    E = A_tmp.copy() 
            # else: E remains A_tmp (no columns selected yet)
        
        # Ensure we have k distinct columns after j_loop for this pass
        # (as per "obtaining an initial solution S with exactly k columns")
        if len(I) < k:
            num_needed = k - len(I)
            # Columns of A_tmp not already in I
            available_fill_cols = [c_idx for c_idx in range(A_tmp.shape[1]) if c_idx not in I]
            
            if len(available_fill_cols) < num_needed: # Not enough unique columns left
                I.update(available_fill_cols) # Add all available
                # print(f"Warning (LSCSS Init t={t_pass}): Could only select {len(I)} distinct columns, k={k}.")
            elif available_fill_cols: # If list is not empty
                I.update(random.sample(available_fill_cols, num_needed))
        
        I_list = list(I) # Make it a list for consistent use

        # 7. If t_pass is 1: Calculate D, update A_tmp to A', reset I
        if t_pass == 1:
            # 8. Calculate alpha for D
            # S1 is B[:, I_list]
            # Numerator is || B - S1 S1_dagger B ||_F
            f_B_S1_val_sq = residual_error(B, I_list)
            
            numerator_for_alpha = np.sqrt(f_B_S1_val_sq) if f_B_S1_val_sq > 0 else 0.0

            try: # (k+1)! can be very large
                factorial_k_plus_1 = math.factorial(k + 1)
            except OverflowError:
                # print(f"Warning: (k+1)! for k={k} overflowed. Alpha might be inaccurate (likely too small/zero).")
                factorial_k_plus_1 = float('inf') # This will make alpha small or zero

            denominator_val_for_alpha_sq = 52 * min(n, d) * factorial_k_plus_1
            
            alpha = 0.0
            if denominator_val_for_alpha_sq > 1e-20 and denominator_val_for_alpha_sq != float('inf'):
                alpha = numerator_for_alpha / np.sqrt(denominator_val_for_alpha_sq)
            elif numerator_for_alpha > 1e-9: # If error is non-trivial but denom is bad
                # print("Warning: Denominator for alpha is zero or inf. Using heuristic small alpha.")
                alpha = 1e-9 # Heuristic small perturbation
            # else alpha remains 0.0 if num_for_alpha is also zero

            diag_len = min(n, d)
            D[np.arange(diag_len), np.arange(diag_len)] = alpha

            # 9. Compute A_prime = B + D
            A_prime = B + D
            A_tmp = A_prime # Update for t_pass=2
            # current_I_indices is implicitly reset at the start of the next t_pass loop
        
        # End of t_pass loop
        if t_pass == 2: # After 2nd pass (selection on A_prime)
            S_indices_for_LS_init_in_A_prime = I_list[:] 

    # --- Local Search Part (Steps 13-15) ---
    # Input to LS is A_prime, k, and S_indices_for_LS_init_in_A_prime

    if A_prime is None: # Should have been set if d > 0
        # This case implies k was 0 or d was 0 initially or some other edge case
        # For safety, if D was not made (e.g. alpha=0, or k makes factorial term huge), A_prime is A
        A_prime = B.copy()
        # And S_indices_for_LS_init_in_A_prime would have been selected on A in pass 2.

    # Ensure S_indices_for_LS_init_in_A_prime has k elements for LS
    # This should already be true due to the fill-up logic in the init loop.
    # But as a safeguard:
    if len(S_indices_for_LS_init_in_A_prime) != k and A_prime.shape[1] > 0 :
        # print(f"Warning: Initial S for LS has {len(S_indices_for_LS_init_in_A_prime)} cols, expected {k}. Re-adjusting.")
        temp_set = set(S_indices_for_LS_init_in_A_prime)
        if len(temp_set) > k:
            S_indices_for_LS_init_in_A_prime = random.sample(list(temp_set), k)
        else: # len < k
            num_needed = k - len(temp_set)
            available_cols = [c_idx for c_idx in range(A_prime.shape[1]) if c_idx not in temp_set]
            if available_cols:
                temp_set.update(random.sample(available_cols, min(num_needed, len(available_cols))))
            S_indices_for_LS_init_in_A_prime = sorted(list(temp_set))


    current_S_indices_in_A_prime = S_indices_for_LS_init_in_A_prime[:]
    
    for iter_ls in range(T):
        if not current_S_indices_in_A_prime and k > 0 and A_prime.shape[1] >= k:
            # print(f"Warning (LS Loop): S became empty. Reinitializing with {k} random cols from A_prime.")
            current_S_indices_in_A_prime = random.sample(range(A_prime.shape[1]), k)
        elif len(current_S_indices_in_A_prime) != k and k > 0 and A_prime.shape[1] >= k :
             # print(f"Warning (LS Loop): S has {len(current_S_indices_in_A_prime)} cols, expected {k}. Reinitializing.")
             current_S_indices_in_A_prime = random.sample(range(A_prime.shape[1]),k)


        current_S_indices_in_A_prime = ls_step(A_prime, k, current_S_indices_in_A_prime)
        # obj_val_on_Aprime = calculate_objective_value(A_prime, current_S_indices_in_A_prime, A_S_source_matrix=A_prime)
        # print(f"LS iter {iter_ls+1}/{T}, current S (on A'): {sorted(current_S_indices_in_A_prime)}, obj (on A'): {obj_val_on_Aprime:.4f}")

    # Final solution: Indices are w.r.t. A_prime, but map directly to B
    # because D is diagonal (doesn't change column indexing).
    return list(set(current_S_indices_in_A_prime))