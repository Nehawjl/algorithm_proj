import numpy as np
import random
import math

from .utility import rel_error, frobenius_norm_sq, residual_and_error, residual_and_error_qr


def get_col_sampling_probs(E_matrix):
    """
    Calculate column sampling probabilities p_i = ||E_{:,i}||²_2 / ||E||²_F
    """
    if E_matrix.shape[1] == 0:
        return np.array([])
    
    col_norms_sq = np.sum(E_matrix**2, axis=0)
    total_norm_sq = np.sum(col_norms_sq)
    if total_norm_sq < 1e-20:
        return np.ones(E_matrix.shape[1]) / E_matrix.shape[1]
    
    probs = col_norms_sq / total_norm_sq

    # Sanity check always passes, consider removing it
    # total_probs = np.sum(probs)
    # assert rel_error(total_probs, 1.0) < 1e-8, f"Error calculating column sampling probabilities, total_probs {total_probs} should be close to 1.0"
    return probs


# --- Algorithm 2: LS (Local Search Step) ---
def ls_step(A_prime, k, current_S_indices):
    """
    Perform one local search step (Algorithm 2)
    """
    # assert len(current_S_indices) == k, f"Current solution indices should have length k {k}"
    _, d = A_prime.shape
    current_S_indices = list(current_S_indices)
    E_residual, f_current = residual_and_error(A_prime, current_S_indices)
    
    # Sample candidate columns
    sampling_probs = get_col_sampling_probs(E_residual)
    if sampling_probs.size == 0:
        return current_S_indices
    
    # num_candidates = min(10 * k, d)  # Avoid sampling more than d columns
    num_candidates = 10 * k
    candidate_indices = np.random.choice(d, size=num_candidates, p=sampling_probs, replace=True)
    
    # Uniformly select swap-in column
    p_swap_in = np.random.choice(candidate_indices)
    
    # Find best swap-out column
    best_indices = current_S_indices
    best_objective = f_current
    for q_idx, q_swap_out in enumerate(current_S_indices):
        if p_swap_in == q_swap_out:
            continue
        new_indices = current_S_indices.copy()
        new_indices[q_idx] = p_swap_in
        # Create new index set
        # assert len(new_indices) == k
        
        f_new = residual_and_error(A_prime, new_indices)[1]
        
        if f_new < best_objective:
            best_objective = f_new
            best_indices = new_indices
    
    return best_indices


# --- Algorithm 1: LSCSS ---
def lscss_algorithm(A, k, T=None):
    """
    LSCSS algorithm implementation
    """
    n, d = A.shape
    
    if k == 0:
        return []
    if k >= d:
        return list(range(d))
    if T is None:
        T = max(1, int(k**2 * np.log(k + 1)))

    B = A.copy()
    # Two-pass initialization
    for t in range(1, 3):
        I_indices = []
        A_current = A if t == 1 else A_prime
        
        # Sample k columns
        for _ in range(k):
            # Compute residual
            E_residual = residual_and_error(A_current, I_indices)[0]
            
            # Sample column
            probs = get_col_sampling_probs(E_residual)
            # assert probs.size > 0, "Sampling probs array should not be none"

            # Exclude already selected columns from sampling
            available_mask = np.ones(d, dtype=bool)
            if I_indices:
                available_mask[I_indices] = False
            # assert np.any(available_mask), "Error sampling for no available column"
            # Renormalize probabilities for available columns
            available_probs = probs[available_mask]
            available_probs = available_probs / np.sum(available_probs)
            available_indices = np.where(available_mask)[0]
            sampled_idx = np.random.choice(available_indices, p=available_probs)
            I_indices.append(sampled_idx)
        
        if t == 1:
            # Compute perturbation matrix D
            residual_norm_sq = residual_and_error(B, I_indices)[1]
            residual_norm = np.sqrt(residual_norm_sq)
            
            # Compute alpha carefully to avoid overflow
            try:
                # log_factorial = (k + 1) * np.log(k + 1) - (k + 1) + 0.5 * np.log(2 * np.pi * (k + 1))
                # log_denominator = 0.5 * (np.log(52) + np.log(min(n, d)) + log_factorial)
                # alpha = residual_norm * np.exp(-log_denominator)
                factorial_term = math.factorial(k + 1)
                denominator = np.sqrt(52 * min(n, d) * factorial_term)
                alpha = residual_norm / denominator if denominator > 0 else 1e-10
            except (OverflowError, ValueError, TypeError):
                # Use Stirling's approximation for large k
                log_factorial = (k + 1) * np.log(k + 1) - (k + 1) + 0.5 * np.log(2 * np.pi * (k + 1))
                log_denominator = 0.5 * (np.log(52) + np.log(min(n, d)) + log_factorial)
                alpha = residual_norm * np.exp(-log_denominator)
            
            # Create diagonal perturbation
            D = np.zeros_like(A)
            diag_size = min(n, d)
            D[np.arange(diag_size), np.arange(diag_size)] = alpha
            
            # Update A for second pass
            A = A + D
            A_prime = B + D
    
    # return I_indices
    S_indices = I_indices
    for i in range(T):
        S_indices = ls_step(A_prime, k, S_indices)
    
    return S_indices


def ls_step_qr(A_prime, k, current_S_indices):
    _, d = A_prime.shape
    current_S_indices = list(current_S_indices)
    E_residual, f_current = residual_and_error_qr(A_prime, current_S_indices)

    sampling_probs = get_col_sampling_probs(E_residual)
    if sampling_probs.size == 0:
        return current_S_indices
    
    num_candidates = 10 * k
    candidate_indices = np.random.choice(d, size=num_candidates, p=sampling_probs, replace=True)
    p_swap_in = np.random.choice(candidate_indices)
    
    best_indices = current_S_indices
    best_objective = f_current
    for q_idx, q_swap_out in enumerate(current_S_indices):
        if p_swap_in == q_swap_out:
            continue
        new_indices = current_S_indices.copy()
        new_indices[q_idx] = p_swap_in
        
        f_new = residual_and_error_qr(A_prime, new_indices)[1]
        
        if f_new < best_objective:
            best_objective = f_new
            best_indices = new_indices
    
    return best_indices


def lscss_algorithm_qr(A, k, T=None):
    n, d = A.shape
    
    if k == 0:
        return []
    if k >= d:
        return list(range(d))
    if T is None:
        T = max(1, int(k**2 * np.log(k + 1)))

    B = A.copy()
    for t in range(1, 3):
        I_indices = []
        A_current = A if t == 1 else A_prime
        
        for _ in range(k):
            E_residual = residual_and_error_qr(A_current, I_indices)[0]
            
            probs = get_col_sampling_probs(E_residual)
            
            available_mask = np.ones(d, dtype=bool)
            if I_indices:
                available_mask[I_indices] = False
            available_probs = probs[available_mask]
            available_probs = available_probs / np.sum(available_probs)
            available_indices = np.where(available_mask)[0]
            sampled_idx = np.random.choice(available_indices, p=available_probs)
            I_indices.append(sampled_idx)
        
        if t == 1:
            residual_norm_sq = residual_and_error_qr(B, I_indices)[1]
            residual_norm = np.sqrt(residual_norm_sq)
            
            try:
                factorial_term = math.factorial(k + 1)
                denominator = np.sqrt(52 * min(n, d) * factorial_term)
                alpha = residual_norm / denominator if denominator > 0 else 1e-10
            except (OverflowError, ValueError, TypeError):
                log_factorial = (k + 1) * np.log(k + 1) - (k + 1) + 0.5 * np.log(2 * np.pi * (k + 1))
                log_denominator = 0.5 * (np.log(52) + np.log(min(n, d)) + log_factorial)
                alpha = residual_norm * np.exp(-log_denominator)
            
            D = np.zeros_like(A)
            diag_size = min(n, d)
            D[np.arange(diag_size), np.arange(diag_size)] = alpha
            
            A = A + D
            A_prime = B + D
    
    S_indices = I_indices
    for i in range(T):
        S_indices = ls_step_qr(A_prime, k, S_indices)
    
    return S_indices