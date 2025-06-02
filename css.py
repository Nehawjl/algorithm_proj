import numpy as np
import math
import random
import time
from sklearn.utils.extmath import randomized_svd # For efficient A_k
import matplotlib.pyplot as plt

# --- Utility Functions ---
def frobenius_norm_sq(matrix):
    return np.linalg.norm(matrix, 'fro')**2

def calculate_objective_value(A_orig, S_indices, A_for_S_cols=None):
    """
    Calculates ||A_orig - S S_dagger A_orig||_F^2
    S_indices: indices of columns selected.
    A_for_S_cols: The matrix from which columns for S are taken (e.g. A_orig or A_prime).
                  If None, assumes A_orig.
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


def get_col_sampling_probs(E_matrix):
    """
    Calculates p_i = ||E_matrix_{:,i}||_2^2 / ||E_matrix||_F^2
    """
    if E_matrix.shape[1] == 0:
        return np.array([])
        
    col_norms_sq = np.sum(np.square(E_matrix), axis=0)
    total_norm_sq = np.sum(col_norms_sq) # This is ||E_matrix||_F^2

    if total_norm_sq < 1e-12: # Avoid division by zero if E_matrix is (close to) zero
        # Uniform probability if matrix is zero, or handle as error
        # print("Warning: Total norm of E_matrix is near zero in get_col_sampling_probs.")
        return np.ones(E_matrix.shape[1]) / E_matrix.shape[1]
    
    return col_norms_sq / total_norm_sq

def get_A_k_error(A_orig, k):
    """
    Computes ||A_orig - A_k||_F^2 using SVD
    A_k is the best rank-k approximation.
    """
    if k == 0:
        return frobenius_norm_sq(A_orig)
    if k >= min(A_orig.shape): # If k is full rank or more
        U, s, Vt = np.linalg.svd(A_orig, full_matrices=False)
        A_k_approx = U @ np.diag(s) @ Vt
        return frobenius_norm_sq(A_orig - A_k_approx)


    # For k < min(A_orig.shape), only sum of squared singular values beyond k
    # ||A - A_k||_F^2 = sum_{i=k+1 to r} sigma_i^2
    # Using randomized_svd for potentially faster computation if k is small
    try:
        # U, Sigma, VT = randomized_svd(A_orig, n_components=min(A_orig.shape), random_state=None)
        # For this error, we only need the singular values
        _, s_all, _ = np.linalg.svd(A_orig, full_matrices=False)
        if k >= len(s_all): # if k is equal or larger than rank
            return 0.0 
        error_A_k = np.sum(s_all[k:]**2)
        return error_A_k
    except Exception as e:
        print(f"SVD for A_k failed: {e}. Falling back to full SVD error calculation.")
        # Fallback if randomized_svd has issues or for exactness
        U, s, Vt = np.linalg.svd(A_orig, full_matrices=True)
        s_full_rank = np.zeros(min(A_orig.shape))
        s_full_rank[:len(s)] = s
        A_k_approx = U[:, :k] @ np.diag(s_full_rank[:k]) @ Vt[:k, :]
        return frobenius_norm_sq(A_orig - A_k_approx)


# --- Algorithm 2: LS (Local Search Step) ---
def ls_algorithm_step(A_prime, k, current_S_indices, A_orig_for_error_calc):
    """
    Performs one step of the local search (Algorithm 2).
    A_prime: The (potentially perturbed) matrix A'.
    k: Number of columns to select.
    current_S_indices: List of column indices currently in S, w.r.t A_prime.
    A_orig_for_error_calc: The original matrix A, used for calculating f(A_orig, S) if needed,
                           but the paper uses f(A', S). We use A_prime for f, as in paper.
    Returns: Updated list of column indices for S.
    """
    n, d = A_prime.shape
    
    # Ensure current_S_indices is a list of unique integers
    # This can be an issue if it's passed as a numpy array that gets reshaped
    current_S_indices = sorted(list(set(int(idx) for idx in np.array(current_S_indices).flatten())))


    if not current_S_indices or len(current_S_indices) != k:
        # This should not happen if initialization is correct
        print(f"Warning: current_S_indices is invalid in LS: {current_S_indices}, k={k}")
        # Fallback: re-initialize S with first k columns or random k columns
        # current_S_indices = list(range(k)) # Or some other robust initialization
        if len(current_S_indices) > k : current_S_indices = current_S_indices[:k]
        while len(current_S_indices) < k and d > len(current_S_indices):
            # Add columns that are not already in current_S_indices
            potential_new_cols = [c for c in range(d) if c not in current_S_indices]
            if not potential_new_cols: break
            current_S_indices.append(random.choice(potential_new_cols))


    current_f_S = calculate_objective_value(A_prime, current_S_indices, A_for_S_cols=A_prime)

    # 1. Compute residual matrix E = A' - S S_dagger A'
    S_matrix = A_prime[:, current_S_indices]
    try:
        S_dagger = np.linalg.pinv(S_matrix)
        E_matrix = A_prime - S_matrix @ S_dagger @ A_prime
    except np.linalg.LinAlgError:
        print("Warning: SVD for S_dagger failed in LS. Using A_prime as E_matrix (high error).")
        E_matrix = A_prime.copy()


    # 2. Sample a set C of 10k column indices from A'
    # Probabilities p_i = ||E_{:,i}||_2^2 / ||E||_F^2
    probs = get_col_sampling_probs(E_matrix)
    if len(probs) == 0 : # d=0 or E_matrix is empty
        return current_S_indices # No columns to sample from

    num_candidates_C = 10 * k
    # Ensure num_candidates_C is not more than available columns
    num_candidates_C = min(num_candidates_C, d) 
    
    # np.random.choice needs 1D array of probabilities
    if probs.ndim > 1: probs = probs.flatten()
    # Handle cases where sum of probs is not 1 (e.g. due to floating point issues)
    probs = probs / np.sum(probs)


    # Sample with replacement. If d < num_candidates_C, it will sample all d columns multiple times.
    # Better to sample without replacement if num_candidates_C > d, or sample min(num_candidates_C, d)
    # However, the paper says "sample each column index i", suggesting sampling with replacement
    # to build a set of size 10k.
    if d > 0:
        candidate_indices_C = np.random.choice(d, size=num_candidates_C, p=probs, replace=True)
    else: # no columns to sample from
        return current_S_indices


    # 3. Uniformly sample an index p from C
    if not candidate_indices_C.size: # If C is empty (e.g. d=0 or k=0)
        return current_S_indices
    p_swap_in_idx = random.choice(candidate_indices_C)

    # 4. Let I be the set of column indices of S in A' (current_S_indices)

    # 5. If there exists q in I such that f(A', A'_{I\{q} U {p}}) < f(A', S)
    best_q_to_swap_out_idx = -1
    min_f_after_swap = current_f_S

    for q_idx_in_S_list, q_col_idx_in_A_prime in enumerate(current_S_indices):
        if q_col_idx_in_A_prime == p_swap_in_idx: # Cannot swap a column with itself if it's already in S
            continue

        # Create new potential S_indices: (I \ {q}) U {p}
        temp_S_indices = current_S_indices[:]
        temp_S_indices.pop(q_idx_in_S_list) # Remove q
        
        # Ensure p is not already in the remaining list before adding
        if p_swap_in_idx not in temp_S_indices:
            temp_S_indices.append(p_swap_in_idx)
        else: # p was already in S (and wasn't q), so this swap effectively removes q
              # This means we are trying to select k-1 columns. This shouldn't happen often
              # if k is maintained. If p is already in S and p != q, then I\{q} U {p} = I\{q}.
              # This case should be handled by ensuring temp_S_indices has k unique columns.
              # For now, let's assume we must maintain k columns. If p is already there,
              # this particular q cannot be swapped for p to maintain k unique columns, unless
              # we allow duplicate columns in S, which is usually not the case.
              # Let's assume S must contain k *distinct* columns.
              # If p is already in (I \ {q}), then adding p again is redundant.
              # The set should naturally handle distinctness.
              pass # p is already in the list (and distinct from q)

        temp_S_indices = sorted(list(set(temp_S_indices))) # Ensure unique and sorted

        if len(temp_S_indices) != k:
            # This can happen if p_swap_in_idx was already in current_S_indices and q was removed.
            # We must select k columns. If p is already in S, this swap means S effectively loses q.
            # The problem is CSS for k columns.
            # For now, if this results in not k columns, skip this swap.
            # Or, it implies that p_swap_in_idx was one of the other columns in S (not q).
            # Then I\{q} U {p} = I\{q}. This would result in k-1 columns.
            # The paper implies A'_{\mathcal{I}\setminus\{q\}\cup\{p\}} still has k columns.
            # This means if p is already in \mathcal{I} and p != q, then this "swap"
            # is just removing q. This is not what a swap usually means in CSS.
            # A robust way:
            potential_S_indices = [idx for idx in current_S_indices if idx != q_col_idx_in_A_prime]
            if p_swap_in_idx not in potential_S_indices:
                 potential_S_indices.append(p_swap_in_idx)
            
            if len(potential_S_indices) != k: # if p was already in I and p!=q, this leads to k-1.
                                              # This means p cannot replace q if p is already in S and distinct from q.
                                              # Or, the problem allows choosing < k cols if p already in S.
                                              # Let's stick to "choose k columns": if p is already in S (and not q),
                                              # then this swap is not meaningful for finding a k-subset.
                continue


            f_val_temp = calculate_objective_value(A_prime, potential_S_indices, A_for_S_cols=A_prime)

            if f_val_temp < min_f_after_swap:
                min_f_after_swap = f_val_temp
                best_q_to_swap_out_idx = q_col_idx_in_A_prime
                # Store the state of potential_S_indices that led to this best swap
                best_potential_S_indices = potential_S_indices 

    # 6 & 7. If improvement found, update S_indices
    if best_q_to_swap_out_idx != -1 and min_f_after_swap < current_f_S:
        # This was the logic from the loop:
        # final_S_indices = [idx for idx in current_S_indices if idx != best_q_to_swap_out_idx]
        # if p_swap_in_idx not in final_S_indices: # Should be true if best_q_to_swap_out_idx != p_swap_in_idx
        #    final_S_indices.append(p_swap_in_idx)
        # current_S_indices = sorted(list(set(final_S_indices)))
        current_S_indices = sorted(list(set(best_potential_S_indices)))


    return current_S_indices


# --- Algorithm 1: LSCSS ---
def lscss_algorithm(A_orig, k, T_local_search_iters):
    """
    Implements the LSCSS algorithm (Algorithm 1).
    A_orig: The original n x d matrix.
    k: Number of columns to select.
    T_local_search_iters: Number of iterations for the local search part.
    """
    n, d = A_orig.shape
    
    if k == 0:
        return [], calculate_objective_value(A_orig, [])
    if k >= d: # Select all columns
        all_indices = list(range(d))
        return all_indices, calculate_objective_value(A_orig, all_indices)

    # For step 9, A is updated. So let's use A_current to track current A.
    A_current = A_orig.copy()
    
    # Store the A_prime that will be used in local search
    A_prime_for_ls = None 
    
    # Store initial S_indices selected on A_prime (end of t=2 loop)
    S_indices_for_ls_init = []

    # 1. Initialize I = empty, E = A, B = A (B not used)
    # Loop for t = 1, 2
    for t_loop in range(1, 3):
        I_current_indices = [] # Reset I for each t_loop if t=1 leads to reset, or before t=2 selection
        
        # Effective A for this loop (A_orig for t=1, A_prime for t=2)
        # E needs to be initialized based on the A for this pass
        # For t=1, E starts as A_orig. For t=2, E starts as A_prime.
        # The paper says E = A - A_I A_I_dagger A.
        # A in this formula should be A_current.
        
        E_matrix = A_current.copy() # Initial E for selecting first column in this pass

        # 3. For j = 1 to k (select k columns)
        for j_loop in range(k):
            if E_matrix.shape[1] == 0 : # No columns left to sample from E_matrix or E_matrix is empty
                 # This case should ideally not be hit if d >= k
                 # If it does, fill I_current_indices with available unique columns if not enough
                 if len(I_current_indices) < k:
                     available_cols = [c for c in range(A_current.shape[1]) if c not in I_current_indices]
                     needed = k - len(I_current_indices)
                     if available_cols:
                         I_current_indices.extend(random.sample(available_cols, min(needed, len(available_cols))))
                 break # Break from j_loop

            probs = get_col_sampling_probs(E_matrix)
            
            # Ensure probs is 1D and sums to 1 for np.random.choice
            if probs.ndim > 1: probs = probs.flatten()
            if not np.isclose(np.sum(probs), 1.0): probs = probs / np.sum(probs)
            if not probs.size: # No columns left in E_matrix to sample (e.g., if E became zero-column)
                # This might happen if E_matrix columns are dropped, but E_matrix should always have d columns.
                # More likely, all probabilities are zero, means E_matrix is zero.
                # In this case, sample uniformly from A_current's columns not yet in I_current_indices.
                available_cols = [c for c in range(A_current.shape[1]) if c not in I_current_indices]
                if not available_cols: break # No more unique columns to pick
                sampled_idx = random.choice(available_cols)
            else:
                # Sample a column index from [d] (indices of A_current, also indices of E_matrix)
                # Need to ensure we don't re-select an already selected column if I should be distinct
                # The paper's "Update I = I U {i}" suggests I is a set of unique indices.
                # So, we sample from available columns.
                
                # Option 1: Sample from all d columns, if already picked, re-sample (less efficient)
                # Option 2: Sample from columns not in I_current_indices, adjusting probabilities.
                # The paper seems to imply sampling from all d columns based on E's column norms.
                # If an already selected column is picked, it doesn't change I.
                # This implies that the number of *distinct* columns might grow slower than j_loop.
                # Let's assume we want k *distinct* columns.
                # The probabilities are for columns of E (which has d columns).
                
                # Simplest interpretation: sample an index i from [d].
                # If i is already in I_current_indices, I_current_indices effectively doesn't change size.
                # This means the inner loop might need more than k iterations to get k distinct columns.
                # Or, "Update I = I U {i}" implies j_loop iterates until |I|=k.
                # The "for j = 1..k" loop structure suggests k additions. Let's assume we add, even if duplicate,
                # then take unique set at the end of j_loop, and if less than k, fill up.
                # Or, more robustly, sample until a new unique column is found.
                
                # Let's go with "sample from [d], then add to set"
                # The probabilities are for columns of E_matrix (which has d columns)
                # The indices are 0 to d-1
                if E_matrix.shape[1] != A_current.shape[1]:
                    # This would be an issue. E should have same number of columns as A_current
                    print(f"Warning: E_matrix column count ({E_matrix.shape[1]}) != A_current column count ({A_current.shape[1]})")
                    # Fallback, recompute E based on A_current and A_current's selected columns
                    if I_current_indices:
                        S_temp = A_current[:, I_current_indices]
                        S_temp_dagger = np.linalg.pinv(S_temp)
                        E_matrix = A_current - S_temp @ S_temp_dagger @ A_current
                    else:
                        E_matrix = A_current.copy()
                    probs = get_col_sampling_probs(E_matrix)
                    if probs.ndim > 1: probs = probs.flatten()
                    if not np.isclose(np.sum(probs), 1.0): probs = probs / np.sum(probs)


                if not probs.any(): # All probabilities are zero (E is zero matrix)
                    # Select a random column not yet chosen
                    available_cols_A = [c_idx for c_idx in range(A_current.shape[1]) if c_idx not in I_current_indices]
                    if not available_cols_A: break # No more unique columns
                    sampled_idx = random.choice(available_cols_A)
                else:
                    # Loop to ensure a new distinct column is added (if desired for k distinct)
                    # However, paper's probability is on E's columns.
                    # Let's stick to sampling from d columns using probs from E.
                    sampled_idx = np.random.choice(A_current.shape[1], p=probs)


            if sampled_idx not in I_current_indices:
                 I_current_indices.append(sampled_idx)
            
            # If after sampling, len(I_current_indices) is still less than j_loop+1
            # (e.g. always sampling already picked ones), we need a robust way to fill.
            # For now, assume this method is fine. If len(I_current_indices) < k at end, fill randomly.
            
            # 5. Update E = A_current - A_I A_I_dagger A_current
            # Ensure I_current_indices is not empty before trying to form S_for_E_calc
            if I_current_indices:
                S_for_E_calc = A_current[:, I_current_indices]
                try:
                    S_for_E_calc_dagger = np.linalg.pinv(S_for_E_calc)
                    E_matrix = A_current - S_for_E_calc @ S_for_E_calc_dagger @ A_current
                except np.linalg.LinAlgError:
                    print(f"Warning: SVD for S_for_E_calc_dagger failed in LSCSS init loop. Using A_current as E.")
                    E_matrix = A_current.copy() # Or A_current - S S_true_inv A_current if S is invertible
            else: # I_current_indices is empty
                E_matrix = A_current.copy()
        
        # Ensure we have k distinct columns after j_loop
        I_current_indices = sorted(list(set(I_current_indices)))
        if len(I_current_indices) < k:
            # print(f"Warning: After init loop for t={t_loop}, only {len(I_current_indices)} distinct columns. Filling randomly.")
            num_needed = k - len(I_current_indices)
            available_cols_A = [c_idx for c_idx in range(A_current.shape[1]) if c_idx not in I_current_indices]
            if len(available_cols_A) < num_needed: # Not enough unique columns in A_current
                I_current_indices.extend(available_cols_A)
            elif available_cols_A : # Check if available_cols_A is not empty
                I_current_indices.extend(random.sample(available_cols_A, num_needed))
            I_current_indices = sorted(list(set(I_current_indices)))


        # 7. If t is 1
        if t_loop == 1:
            # 8. Initialize D, calculate alpha
            # S1 is A_orig[:, I_current_indices]
            if not I_current_indices: # Should have k indices now
                 # Fallback if I_current_indices is empty (e.g., k=0 or d=0 initially)
                 S1_error_norm_F = frobenius_norm_sq(A_orig)
            else:
                S1_error_norm_F = calculate_objective_value(A_orig, I_current_indices, A_for_S_cols=A_orig)

            # Denominator for alpha
            # (k+1)! can be huge. Let's use log gamma for factorial if k is large, or cap k.
            # Max k for float64 factorial is around 170. For this problem, k is likely much smaller.
            # Let's assume k is small enough for math.factorial.
            if k + 1 > 20: # Heuristic threshold
                # print(f"Warning: (k+1)! with k={k} might be very large or overflow. Alpha might be inaccurate.")
                # Using a very large number for factorial term if it would overflow, making alpha very small
                log_factorial_k_plus_1 = math.lgamma(k + 2) 
                # log_denominator_factor = log_factorial_k_plus_1 + 0.5 * np.log(52 * min(n, d)) # error in formula, it's ( (denom)^0.5 )
                # So it's 0.5 * (log(52) + log(min(n,d)) + log((k+1)!))
                # This term is ( (constant) * (k+1)! )^1/2
                # Let's compute (k+1)! directly, catch overflow
                try:
                    factorial_term = math.factorial(k+1)
                except OverflowError:
                    factorial_term = float('inf') # Makes alpha effectively zero if S1_error_norm_F is finite
            else:
                factorial_term = math.factorial(k+1)

            denominator_val_for_alpha_sq = 52 * min(n, d) * factorial_term
            
            if denominator_val_for_alpha_sq < 1e-12 or denominator_val_for_alpha_sq == float('inf'):
                alpha = 1e-9 # A very small perturbation if denominator is problematic
                # print("Warning: Denominator for alpha is zero or inf. Using small alpha.")
            else:
                alpha = np.sqrt(S1_error_norm_F) / np.sqrt(denominator_val_for_alpha_sq) # sqrt(error_F_sq) is error_F
                                                                                       # paper has ||A - S1S1dag A||_F / (denom)^1/2
                                                                                       # My S1_error_norm_F is already squared.
                alpha = np.sqrt(S1_error_norm_F) / (denominator_val_for_alpha_sq**0.5) if S1_error_norm_F >=0 else 0
                # No, paper is correct: (||A - S S_dag A||_F) / (CONST)^1/2. My S1_error_norm_F is ||...||_F^2
                # So it should be sqrt(S1_error_norm_F) / (CONST)^1/2
                # Or S1_error_norm_F / (CONST) if alpha was D_ii^2.
                # Text: D_ii = ||...||_F / (CONST)^1/2. My S1_error_norm_F is ||...||_F^2.
                # So D_ii = sqrt(S1_error_norm_F) / (CONST)^1/2
                alpha = np.sqrt(S1_error_norm_F) / (denominator_val_for_alpha_sq**0.5) if S1_error_norm_F > 0 and denominator_val_for_alpha_sq > 0 else 1e-9


            D_matrix = np.zeros_like(A_orig)
            diag_len = min(n, d)
            D_matrix[np.arange(diag_len), np.arange(diag_len)] = alpha

            # 9. Compute A_prime = A_orig + D, and set I = empty
            A_prime_for_ls = A_orig + D_matrix # This is A'
            A_current = A_prime_for_ls # Update A_current for the t=2 loop
            # I_current_indices = [] # Reset for next selection round on A_prime
            # This reset happens at the start of the t_loop anyway.
        
        # End of t_loop
        if t_loop == 2: # After 2nd loop (on A_prime)
            S_indices_for_ls_init = I_current_indices[:] # These are indices for A_prime

    # Local Search Part (Steps 13-15 conceptually)
    # Input to LS is A_prime_for_ls, k, and S_indices_for_ls_init
    if A_prime_for_ls is None: # Should be set if d > 0
        A_prime_for_ls = A_orig.copy() # Fallback if D was not constructed
        # S_indices_for_ls_init should also be from A_orig in this case
        # (This situation might occur if k=0 or d=0, handled earlier, or if t_loop logic error)

    current_S_final_indices = S_indices_for_ls_init[:]
    
    # Ensure k columns for LS, if S_indices_for_ls_init is not k (e.g. if k > d_perturbed)
    if len(current_S_final_indices) != k and A_prime_for_ls.shape[1] > 0:
        # print(f"Warning: Initial S for LS has {len(current_S_final_indices)} cols, expected {k}. Adjusting.")
        if len(current_S_final_indices) > k:
            current_S_final_indices = current_S_final_indices[:k]
        else:
            num_needed = k - len(current_S_final_indices)
            available_cols_A_prime = [c_idx for c_idx in range(A_prime_for_ls.shape[1]) if c_idx not in current_S_final_indices]
            if available_cols_A_prime :
                 current_S_final_indices.extend(random.sample(available_cols_A_prime, min(num_needed, len(available_cols_A_prime))))
            current_S_final_indices = sorted(list(set(current_S_final_indices)))


    for iter_ls in range(T_local_search_iters):
        if not current_S_final_indices and k > 0: # If S becomes empty somehow
            # print("Warning: S became empty during LS. Reinitializing S for LS.")
            current_S_final_indices = random.sample(range(A_prime_for_ls.shape[1]), k) # Random restart for S

        current_S_final_indices = ls_algorithm_step(A_prime_for_ls, k, current_S_final_indices, A_orig)
        # Optional: print(f"LS iter {iter_ls+1}, error on A_prime: {calculate_objective_value(A_prime_for_ls, current_S_final_indices, A_prime_for_ls)}")


    # Final solution S consists of columns from A_orig corresponding to indices in current_S_final_indices
    # (Indices are w.r.t A_prime, but D is diagonal, so indices map directly to A_orig)
    final_objective_val = calculate_objective_value(A_orig, current_S_final_indices, A_for_S_cols=A_orig)
    
    return sorted(list(set(current_S_final_indices))), final_objective_val


# --- Baseline Algorithms ---

def random_css(A_orig, k):
    n, d = A_orig.shape
    if k >= d:
        selected_indices = list(range(d))
    else:
        selected_indices = random.sample(range(d), k)
    
    error = calculate_objective_value(A_orig, selected_indices)
    return sorted(selected_indices), error

def greedy_css(A_orig, k):
    """
    A common greedy approach: iteratively add the column that maximally reduces the current error.
    This is computationally intensive due to repeated pinv.
    """
    n, d = A_orig.shape
    if k == 0: return [], frobenius_norm_sq(A_orig)
    if k >= d: 
        all_indices = list(range(d))
        return all_indices, calculate_objective_value(A_orig, all_indices)

    selected_indices = []
    remaining_indices = list(range(d))
    
    current_A_for_S = A_orig # Columns are from A_orig

    for _ in range(k):
        if not remaining_indices: break # No more columns to select

        best_new_idx = -1
        min_error_for_this_step = float('inf')

        for candidate_idx in remaining_indices:
            temp_selection = selected_indices + [candidate_idx]
            error = calculate_objective_value(A_orig, temp_selection, A_for_S_cols=current_A_for_S)
            
            if error < min_error_for_this_step:
                min_error_for_this_step = error
                best_new_idx = candidate_idx
        
        if best_new_idx != -1:
            selected_indices.append(best_new_idx)
            remaining_indices.remove(best_new_idx)
        else: # No column improved error (or remaining_indices was empty)
            break 
            
    final_error = calculate_objective_value(A_orig, selected_indices, A_for_S_cols=current_A_for_S)
    return sorted(selected_indices), final_error

# --- Main Experimentation Logic ---
if __name__ == '__main__':
    # --- Configuration ---
    N_ROWS = 200
    D_COLS = 100
    K_SELECT = 5
    T_LS_ITERS_PAPER_EXPERIMENT = 2 * K_SELECT # As per paper's experimental setup for ILS/LSCSS
    T_LS_ITERS_THEORY = K_SELECT**2 * int(np.log2(K_SELECT) + 1) if K_SELECT > 1 else K_SELECT**2 # Heuristic for O(k^2 log k)

    # Use T_LS_ITERS_PAPER_EXPERIMENT for faster runs similar to paper's comparison setting
    T_LSCSS = T_LS_ITERS_PAPER_EXPERIMENT
    
    # --- Generate Synthetic Data ---
    print(f"Generating synthetic data: N={N_ROWS}, D={D_COLS}, K={K_SELECT}")
    # A = np.random.rand(N_ROWS, D_COLS)
    
    # Structured data (e.g., low-rank + noise) might be more interesting
    rank = max(1, D_COLS // 2) # Make it somewhat low-rank
    U_true = np.random.rand(N_ROWS, rank)
    V_true = np.random.rand(rank, D_COLS)
    A_low_rank = U_true @ V_true
    A_noise = np.random.rand(N_ROWS, D_COLS) * 0.1 # Add some noise
    A = A_low_rank + A_noise
    print("Data generated.")

    # --- Calculate Optimal A_k error (benchmark) ---
    optimal_A_k_error = get_A_k_error(A, K_SELECT)
    print(f"Optimal SVD error ||A - A_k||_F^2: {optimal_A_k_error:.4f}")
    if optimal_A_k_error < 1e-9: # if A_k is almost perfect
        print("Warning: Optimal A_k error is very small. Error ratios might be unstable.")
        # optimal_A_k_error = 1e-9 # Avoid division by zero for ratio

    results = {}

    # --- Run LSCSS ---
    print(f"\nRunning LSCSS with T={T_LSCSS}...")
    start_time = time.time()
    lscss_indices, lscss_error = lscss_algorithm(A, K_SELECT, T_LSCSS)
    lscss_time = time.time() - start_time
    results['LSCSS'] = {'indices': lscss_indices, 'error': lscss_error, 'time': lscss_time,
                        'ratio': lscss_error / optimal_A_k_error if optimal_A_k_error > 1e-9 else float('inf')}
    print(f"LSCSS selected: {lscss_indices}")
    print(f"LSCSS error: {lscss_error:.4f}, Ratio: {results['LSCSS']['ratio']:.4f}, Time: {lscss_time:.4f}s")

    # --- Run Random CSS ---
    print("\nRunning Random CSS...")
    start_time = time.time()
    random_indices, random_error = random_css(A, K_SELECT)
    random_time = time.time() - start_time
    results['Random'] = {'indices': random_indices, 'error': random_error, 'time': random_time,
                         'ratio': random_error / optimal_A_k_error if optimal_A_k_error > 1e-9 else float('inf')}
    print(f"Random CSS selected: {random_indices}")
    print(f"Random CSS error: {random_error:.4f}, Ratio: {results['Random']['ratio']:.4f}, Time: {random_time:.4f}s")

    # --- Run Greedy CSS ---
    print("\nRunning Greedy CSS...")
    start_time = time.time()
    greedy_indices, greedy_error = greedy_css(A, K_SELECT)
    greedy_time = time.time() - start_time
    results['Greedy'] = {'indices': greedy_indices, 'error': greedy_error, 'time': greedy_time,
                         'ratio': greedy_error / optimal_A_k_error if optimal_A_k_error > 1e-9 else float('inf')}
    print(f"Greedy CSS selected: {greedy_indices}")
    print(f"Greedy CSS error: {greedy_error:.4f}, Ratio: {results['Greedy']['ratio']:.4f}, Time: {greedy_time:.4f}s")
    
    print("\n--- Summary ---")
    print(f"Optimal SVD error ||A - A_k||_F^2: {optimal_A_k_error:.4f}")
    for alg_name, res in results.items():
        print(f"{alg_name}: Error Ratio = {res['ratio']:.4f}, Error = {res['error']:.4f}, Time = {res['time']:.4f}s")

    # --- Correctness Validation Ideas (Conceptual) ---
    # 1. Small examples:
    #    A_test = np.array([[1,0,0,10], [0,1,0,11], [0,0,1,12]]) , k=1. Expected: col 3 (index 3)
    #    A_test = np.array([[1,10,0], [1,10,0], [0,0,1]]), k=1. Expected: col 1 (index 1)
    #    A_test = np.array([[1,0,5,0],[0,1,5,0],[0,0,0,1]]), k=2. Expected: [2,3] or [0,2] or [1,2] (cols with 5 and 1)
    #    Run algorithms on these and check selected indices and error.
    
    # 2. Monotonicity for Greedy: Error should generally not increase.
    # 3. LSCSS convergence: Track error within the T_LSCSS loop of lscss_algorithm.
    #    Modify lscss_algorithm to store errors at each LS step and plot them.
    #    `ls_errors_over_time = []`
    #    `ls_errors_over_time.append(calculate_objective_value(A_prime_for_ls, current_S_final_indices, A_prime_for_ls))`
    #    Then plot `ls_errors_over_time`.

    # --- Visualization ---
    labels = list(results.keys())
    error_ratios = [results[label]['ratio'] for label in labels]
    times = [results[label]['time'] for label in labels]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].bar(labels, error_ratios, color=['blue', 'orange', 'green'])
    axs[0].set_ylabel('Error Ratio (||A-SS_daggerA||_F^2 / ||A-A_k||_F^2)')
    axs[0].set_title('Algorithm Performance (Error Ratio)')
    axs[0].set_ylim(bottom=0.9, top=max(2.0, max(error_ratios)*1.1) if any(r > 0 for r in error_ratios) else 2.0) # Start y-axis near 1.0 for better comparison

    axs[1].bar(labels, times, color=['blue', 'orange', 'green'])
    axs[1].set_ylabel('Running Time (seconds)')
    axs[1].set_title('Algorithm Running Time')
    axs[1].set_yscale('log') # Times can vary a lot

    plt.tight_layout()
    plt.show()

    # For convergence plot (needs modification in lscss_algorithm to collect errors):
    # plt.figure()
    # plt.plot(range(len(ls_errors_over_time)), ls_errors_over_time)
    # plt.xlabel("LS Iteration")
    # plt.ylabel("Error on A_prime (||A' - SS_daggerA'||_F^2)")
    # plt.title("LSCSS Local Search Convergence")
    # plt.show()

    # For varying k plot (run the whole experiment for different k values):
    # k_values = [2, 3, 4, 5, 6]
    # lscss_ratios_vs_k = []
    # greedy_ratios_vs_k = []
    # ... run experiments for each k ...
    # plt.figure()
    # plt.plot(k_values, lscss_ratios_vs_k, marker='o', label='LSCSS')
    # plt.plot(k_values, greedy_ratios_vs_k, marker='x', label='Greedy')
    # plt.xlabel("k (Number of selected columns)")
    # plt.ylabel("Error Ratio")
    # plt.title("Error Ratio vs. k")
    # plt.legend()
    # plt.show()