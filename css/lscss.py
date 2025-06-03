import numpy as np
import random
import math

from .utility import residual_and_error, rel_error, residual, residual_error, frobenius_norm_sq


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
    
    for q_swap_out in current_S_indices:
        # Create new index set
        new_indices = set(current_S_indices)
        new_indices.remove(q_swap_out)
        new_indices.add(p_swap_in)
        # assert len(new_indices) == k
        
        new_indices_list = list(new_indices)
        f_new = residual_error(A_prime, new_indices_list)
        
        if f_new < best_objective:
            best_objective = f_new
            best_indices = new_indices_list
    
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
            E_residual = residual(A_current, I_indices)
            
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
            residual_norm_sq = residual_error(B, I_indices)
            residual_norm = np.sqrt(residual_norm_sq)
            
            # Compute alpha carefully to avoid overflow
            try:
                # log_factorial = (k + 1) * np.log(k + 1) - (k + 1) + 0.5 * np.log(2 * np.pi * (k + 1))
                # log_denominator = 0.5 * (np.log(52) + np.log(min(n, d)) + log_factorial)
                # alpha = residual_norm * np.exp(-log_denominator)
                factorial_term = math.factorial(k + 1)
                denominator = np.sqrt(52 * min(n, d) * factorial_term)
                alpha = residual_norm / denominator if denominator > 0 else 1e-10
            except (OverflowError, ValueError):
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


def ls_step_incremental(A_prime, k, current_S_indices):
    """
    执行局部搜索步骤，使用Sherman-Morrison-Woodbury公式增量更新残差
    """
    n, d = A_prime.shape
    current_S_indices = list(current_S_indices)  # 确保是列表类型
    
    # 计算当前解的伪逆映射和残差
    S = A_prime[:, current_S_indices]
    # 预计算当前解的S_pinv_A，避免重复计算
    S_pinv_A = np.linalg.lstsq(S, A_prime, rcond=None)[0]
    E_residual = A_prime - S @ S_pinv_A
    f_current = frobenius_norm_sq(E_residual)
    
    # 采样候选列
    sampling_probs = get_col_sampling_probs(E_residual)
    if sampling_probs.size == 0:
        return current_S_indices
    
    num_candidates = 10 * k
    candidate_indices = np.random.choice(d, size=num_candidates, p=sampling_probs, replace=True)
    
    # 均匀选择swap-in列
    p_swap_in = np.random.choice(candidate_indices)
    
    # 找到最佳swap-out列
    best_indices = current_S_indices
    best_objective = f_current
    best_S_pinv_A = S_pinv_A
    
    for q_idx, q_swap_out in enumerate(current_S_indices):
        # 如果swap-in和swap-out相同，跳过
        if p_swap_in == q_swap_out:
            continue
            
        # 创建新索引集
        new_indices = current_S_indices.copy()
        new_indices[q_idx] = p_swap_in
        
        # 使用Sherman-Morrison-Woodbury公式增量更新
        a_p = A_prime[:, p_swap_in].reshape(-1, 1)  # 新列
        a_q = A_prime[:, q_swap_out].reshape(-1, 1)  # 旧列
        
        # 计算差异向量和更新系数
        delta = a_p - a_q  # 列差异
        
        # 创建单位向量e_q，只有q_idx位置为1
        e_q = np.zeros((k, 1))
        e_q[q_idx] = 1
        
        # 计算中间项
        S_pinv_delta = S_pinv_A[:, p_swap_in].reshape(-1, 1) - e_q
        
        # 使用SMW公式计算更新
        try:
            # 计算分母，避免除以零
            denominator = 1 + e_q.T @ S_pinv_delta
            if abs(denominator) < 1e-10:
                # 分母太小，回退到直接计算
                new_S = A_prime[:, new_indices]
                new_S_pinv_A = np.linalg.lstsq(new_S, A_prime, rcond=None)[0]
                new_residual = A_prime - new_S @ new_S_pinv_A
                f_new = frobenius_norm_sq(new_residual)
            else:
                # 使用SMW公式更新S_pinv_A
                update_term = (S_pinv_delta @ e_q.T) / denominator
                new_S_pinv_A = S_pinv_A - update_term
                
                # 计算新残差和目标函数值
                new_S = A_prime[:, new_indices]
                new_residual = A_prime - new_S @ new_S_pinv_A
                f_new = frobenius_norm_sq(new_residual)
        except:
            # 如果更新公式出错，回退到直接计算
            new_S = A_prime[:, new_indices]
            new_S_pinv_A = np.linalg.lstsq(new_S, A_prime, rcond=None)[0]
            new_residual = A_prime - new_S @ new_S_pinv_A
            f_new = frobenius_norm_sq(new_residual)
        
        # 更新最佳解
        if f_new < best_objective:
            best_objective = f_new
            best_indices = new_indices
            best_S_pinv_A = new_S_pinv_A
    
    return best_indices


def lscss_algorithm_incremental(A, k, T=None):
    """
    使用增量更新的LSCSS算法实现
    """
    # 前面的初始化代码与原算法相同，直到局部搜索部分
    n, d = A.shape
    
    if k == 0:
        return []
    if k >= d:
        return list(range(d))
    if T is None:
        T = max(1, int(k**2 * np.log(k + 1)))

    B = A.copy()
    # 两轮初始化
    for t in range(1, 3):
        I_indices = []
        A_current = A if t == 1 else A_prime
        
        # 采样k列
        for _ in range(k):
            # 计算残差
            E_residual, _ = residual_and_error(A_current, I_indices)
            
            # 采样列
            probs = get_col_sampling_probs(E_residual)
            
            # 排除已选择的列
            available_mask = np.ones(d, dtype=bool)
            if I_indices:
                available_mask[I_indices] = False
            
            # 重新归一化可用列的概率
            available_probs = probs[available_mask]
            available_probs = available_probs / np.sum(available_probs)
            available_indices = np.where(available_mask)[0]
            sampled_idx = np.random.choice(available_indices, p=available_probs)
            I_indices.append(sampled_idx)
        
        if t == 1:
            # 计算扰动矩阵D
            _, residual_norm_sq = residual_and_error(B, I_indices)
            residual_norm = np.sqrt(residual_norm_sq)
            
            # 计算alpha，避免溢出
            try:
                factorial_term = math.factorial(k + 1)
                denominator = np.sqrt(52 * min(n, d) * factorial_term)
                alpha = residual_norm / denominator if denominator > 0 else 1e-10
            except (OverflowError, ValueError):
                # 使用Stirling近似公式处理大k值
                log_factorial = (k + 1) * np.log(k + 1) - (k + 1) + 0.5 * np.log(2 * np.pi * (k + 1))
                log_denominator = 0.5 * (np.log(52) + np.log(min(n, d)) + log_factorial)
                alpha = residual_norm * np.exp(-log_denominator)
            
            # 创建对角扰动
            D = np.zeros_like(A)
            diag_size = min(n, d)
            D[np.arange(diag_size), np.arange(diag_size)] = alpha
            
            # 更新A进行第二轮计算
            A = A + D
            A_prime = B + D
    
    # 局部搜索阶段使用增量更新
    S_indices = I_indices
    for i in range(T):
        S_indices = ls_step_incremental(A_prime, k, S_indices)
    
    return S_indices


def residual_error_qr(matrix, indices):
    """
    使用QR分解计算残差误差，避免SVD计算
    """
    A, I = matrix, indices
    if not I:
        return frobenius_norm_sq(A)
    
    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()
    
    S = A[:, I]
    if S.shape[1] == 0:
        return frobenius_norm_sq(A)
    
    m, n = S.shape  # m行n列
    
    try:
        if m >= n:  # 超定或正定系统
            # 使用QR分解进行计算
            Q, R = np.linalg.qr(S, mode='reduced')
            # 使用R求解系统方程，避免直接计算逆矩阵
            S_pinv_A = np.linalg.solve(R, Q.T @ A)
            residual = A - S @ S_pinv_A
            error_val = frobenius_norm_sq(residual)
        else:  # 欠定系统，使用lstsq
            S_pinv_A = np.linalg.lstsq(S, A, rcond=None)[0]
            residual = A - S @ S_pinv_A
            error_val = frobenius_norm_sq(residual)
    except np.linalg.LinAlgError:
        print(f"Warning: QR decomposition failed for S with columns {I}. Assigning high error.")
        error_val = float('inf')
    
    return error_val


def residual_and_error_qr(matrix, indices):
    """
    使用QR分解计算残差和误差，返回残差矩阵和误差值
    """
    A, I = matrix, indices
    if not I:
        return A.copy(), frobenius_norm_sq(A)
    
    if not isinstance(I, (list, np.ndarray)) or (isinstance(I, np.ndarray) and I.ndim > 1):
        I = np.array(I).flatten().tolist()
    
    S = A[:, I]
    if S.shape[1] == 0:
        return A.copy(), frobenius_norm_sq(A)
    
    m, n = S.shape  # m行n列
    
    try:
        if m >= n:  # 超定或正定系统
            # 使用QR分解进行计算
            Q, R = np.linalg.qr(S, mode='reduced')
            # 使用R求解系统方程，避免直接计算逆矩阵
            S_pinv_A = np.linalg.solve(R, Q.T @ A)
            residual = A - S @ S_pinv_A
            error_val = frobenius_norm_sq(residual)
        else:  # 欠定系统，使用lstsq
            S_pinv_A = np.linalg.lstsq(S, A, rcond=None)[0]
            residual = A - S @ S_pinv_A
            error_val = frobenius_norm_sq(residual)
    except np.linalg.LinAlgError:
        print(f"Warning: QR decomposition failed for S with columns {I}. Assigning high error.")
        residual = A.copy()
        error_val = float('inf')
    
    return residual, error_val


def ls_step_qr(A_prime, k, current_S_indices):
    """
    使用QR分解优化的局部搜索步骤
    """
    n, d = A_prime.shape
    current_S_indices = list(current_S_indices)
    
    # 使用QR分解计算残差
    E_residual, f_current = residual_and_error_qr(A_prime, current_S_indices)
    
    # 采样候选列
    sampling_probs = get_col_sampling_probs(E_residual)
    if sampling_probs.size == 0:
        return current_S_indices
    
    # 采样10k个候选列
    num_candidates = 10 * k
    candidate_indices = np.random.choice(d, size=num_candidates, p=sampling_probs, replace=True)
    
    # 均匀选择swap-in列
    p_swap_in = np.random.choice(candidate_indices)
    
    # 找到最佳swap-out列
    best_indices = current_S_indices
    best_objective = f_current
    
    for q_idx, q_swap_out in enumerate(current_S_indices):
        # 跳过相同的列
        if p_swap_in == q_swap_out:
            continue
            
        # 创建新索引集
        new_indices = current_S_indices.copy()
        new_indices[q_idx] = p_swap_in
        
        # 使用QR分解计算新解的残差误差
        f_new = residual_error_qr(A_prime, new_indices)
        
        if f_new < best_objective:
            best_objective = f_new
            best_indices = new_indices
    
    return best_indices


def lscss_algorithm_qr(A, k, T=None):
    """
    使用QR分解优化的LSCSS算法实现
    """
    n, d = A.shape
    
    if k == 0:
        return []
    if k >= d:
        return list(range(d))
    if T is None:
        T = max(1, int(k**2 * np.log(k + 1)))

    B = A.copy()
    # 两轮初始化
    for t in range(1, 3):
        I_indices = []
        A_current = A if t == 1 else A_prime
        
        # 采样k列
        for _ in range(k):
            # 计算残差
            E_residual, _ = residual_and_error_qr(A_current, I_indices)
            
            # 采样列
            probs = get_col_sampling_probs(E_residual)
            
            # 排除已选择的列
            available_mask = np.ones(d, dtype=bool)
            if I_indices:
                available_mask[I_indices] = False
            
            # 重新归一化概率
            available_probs = probs[available_mask]
            if np.sum(available_probs) > 0:
                available_probs = available_probs / np.sum(available_probs)
                available_indices = np.where(available_mask)[0]
                sampled_idx = np.random.choice(available_indices, p=available_probs)
                I_indices.append(sampled_idx)
            else:
                # 如果所有列残差都接近零，随机选择
                available_indices = np.where(available_mask)[0]
                if len(available_indices) > 0:
                    sampled_idx = np.random.choice(available_indices)
                    I_indices.append(sampled_idx)
                else:
                    break
        
        if t == 1:
            # 计算扰动矩阵D
            _, residual_norm_sq = residual_and_error_qr(B, I_indices)
            residual_norm = np.sqrt(residual_norm_sq)
            
            # 计算alpha，避免溢出
            try:
                factorial_term = math.factorial(k + 1)
                denominator = np.sqrt(52 * min(n, d) * factorial_term)
                alpha = residual_norm / denominator if denominator > 0 else 1e-10
            except (OverflowError, ValueError):
                # 使用Stirling近似处理大k值
                log_factorial = (k + 1) * np.log(k + 1) - (k + 1) + 0.5 * np.log(2 * np.pi * (k + 1))
                log_denominator = 0.5 * (np.log(52) + np.log(min(n, d)) + log_factorial)
                alpha = residual_norm * np.exp(-log_denominator)
            
            # 创建对角扰动
            D = np.zeros_like(A)
            diag_size = min(n, d)
            D[np.arange(diag_size), np.arange(diag_size)] = alpha
            
            # 更新A进行第二轮计算
            A = A + D
            A_prime = B + D
    
    # 局部搜索阶段使用QR分解优化
    S_indices = I_indices
    for i in range(T):
        S_indices = ls_step_qr(A_prime, k, S_indices)
    
    return S_indices