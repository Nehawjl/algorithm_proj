import numpy as np
import math
from scipy.linalg import pinv # 更稳定的伪逆
import pandas as pd

def frobenius_norm_squared(matrix):
    """计算矩阵的 Frobenius 范数的平方"""
    return np.linalg.norm(matrix, 'fro')**2

def calculate_residual_error_squared(A_source_matrix, S_selected_cols_matrix):
    """
    计算残差 ||A_source_matrix - S S† A_source_matrix||_F^2
    A_source_matrix: 源矩阵 (可以是原始 A 或扰动后的 A')，n x d
    S_selected_cols_matrix: 选择的 k 列从 A_source_matrix 中构成的矩阵，n x k
    """
    if S_selected_cols_matrix is None or S_selected_cols_matrix.shape[1] == 0:
        return frobenius_norm_squared(A_source_matrix)
    
    # 检查 S_selected_cols_matrix 的列数是否为 k (或者至少是预期的列数)
    # 并且它的秩是否足够。如果秩小于列数，伪逆仍可计算，但投影可能不是我们期望的。
    # CSS 目标是选出 k 列。
    if S_selected_cols_matrix.shape[1] < 1: # 没有选出列
        return frobenius_norm_squared(A_source_matrix)

    try:
        S_pinv = pinv(S_selected_cols_matrix)
    except np.linalg.LinAlgError:
        print(f"警告: 在 calculate_residual_error_squared 中计算伪逆失败。S 形状: {S_selected_cols_matrix.shape}")
        # 返回一个非常大的错误值，表示这个选择非常差
        return float('inf')

    projection_onto_S_space = S_selected_cols_matrix @ S_pinv
    A_projected = projection_onto_S_space @ A_source_matrix
    residual_matrix = A_source_matrix - A_projected
    return frobenius_norm_squared(residual_matrix)

def select_columns_for_initialization_or_LS(matrix_to_select_from, k_cols_to_select, matrix_for_E_calc):
    """
    实现了 Algorithm 1 的 steps 3-6 (在 for t 循环内部的部分)。
    这个函数用于:
    1. (t=1时) 从 A_orig 中选择 k 列用于计算 alpha。
       此时 matrix_to_select_from = A_orig, matrix_for_E_calc = A_orig.
    2. (t=2时) 从 A_prime 中选择 k 列作为局部搜索的初始解。
       此时 matrix_to_select_from = A_prime, matrix_for_E_calc = A_prime.

    matrix_to_select_from:  从中实际“挑选”列形成 A_I 的矩阵 (A_orig 或 A_prime)。
    k_cols_to_select:       要选择的列数 k。
    matrix_for_E_calc:      用于计算 E = A - A_I A_I† A 公式的那个 "A" (A_orig 或 A_prime)。
    """
    n, d = matrix_to_select_from.shape
    selected_indices = []
    
    # E_current 是基于 matrix_for_E_calc 进行计算的。
    # 初始时 E = matrix_for_E_calc
    E_current_residual_matrix = np.copy(matrix_for_E_calc) 
    
    available_indices = list(range(d)) # 这些索引是相对于 matrix_to_select_from 的

    for _ in range(k_cols_to_select):
        if not available_indices:
            break 

        col_norms_sq_in_E = []
        # 对 E_current_residual_matrix 的每一列（对应于 matrix_to_select_from 的可用列）计算范数
        # 注意：E_current_residual_matrix 的维度与 matrix_for_E_calc 相同。
        # 我们需要用 available_indices 来索引 E_current_residual_matrix 的列。
        for col_idx_in_available in available_indices:
            # ||E_{:i}||_F^2, 其中 i 是 matrix_to_select_from 中的一个可用列索引
            # E_current_residual_matrix 的第 col_idx_in_available 列
            col_norm_sq = frobenius_norm_squared(E_current_residual_matrix[:, [col_idx_in_available]])
            col_norms_sq_in_E.append(col_norm_sq)
        
        col_norms_sq_in_E = np.array(col_norms_sq_in_E)
        
        if np.sum(col_norms_sq_in_E) == 0:
            if not available_indices: break # 无列可选
            probabilities = np.ones(len(available_indices)) / len(available_indices)
        else:
            probabilities = col_norms_sq_in_E / np.sum(col_norms_sq_in_E)
        
        # 确保概率和为1
        probabilities /= np.sum(probabilities)
        
        chosen_relative_idx = np.random.choice(len(available_indices), p=probabilities)
        chosen_actual_idx = available_indices.pop(chosen_relative_idx) 
        selected_indices.append(chosen_actual_idx)

        # 更新 E: E = A_for_E_calc - (A_I_from_select_from) * (A_I_from_select_from)† * A_for_E_calc
        if selected_indices:
            # A_I 是从 matrix_to_select_from 中根据 selected_indices 选出来的列
            A_I_selected = matrix_to_select_from[:, selected_indices]
            
            if A_I_selected.shape[1] == 0: continue # 不应发生

            try:
                A_I_pinv = pinv(A_I_selected)
            except np.linalg.LinAlgError:
                print(f"警告: 在 E 更新时计算伪逆失败。A_I 形状: {A_I_selected.shape}")
                break 

            projection_matrix_onto_A_I_space = A_I_selected @ A_I_pinv
            # E 是通过将 matrix_for_E_calc 投影到 A_I 的列空间，然后取残差来更新的
            E_current_residual_matrix = matrix_for_E_calc - projection_matrix_onto_A_I_space @ matrix_for_E_calc
            
    return sorted(selected_indices)


def local_search_step_LS(A_prime_matrix, k_target_cols, S_current_col_indices_in_A_prime):
    """
    Algorithm 2: LS (Local Search step)
    A_prime_matrix: 扰动后的 n x d 矩阵 (论文中用 A')
    k_target_cols: 要选择的列数 k
    S_current_col_indices_in_A_prime: 当前解 S 中列在 A_prime_matrix 中的索引列表
    """
    n_prime, d_prime = A_prime_matrix.shape
    
    # 当前解 S 对应的矩阵
    S_current_matrix_from_A_prime = A_prime_matrix[:, S_current_col_indices_in_A_prime]

    # 1. 计算残差矩阵 E = A' - SS†A'
    current_total_residual_sq = calculate_residual_error_squared(A_prime_matrix, S_current_matrix_from_A_prime)
    
    # 为了计算 E 的列范数，我们需要 E 本身
    try:
        S_current_pinv = pinv(S_current_matrix_from_A_prime)
    except np.linalg.LinAlgError:
        print(f"警告: 在 local_search_step_LS 中计算 S_current_pinv 失败。形状: {S_current_matrix_from_A_prime.shape}")
        return S_current_col_indices_in_A_prime # 无法进行，不改变

    projection_onto_S_current = S_current_matrix_from_A_prime @ S_current_pinv
    E_ls_residual_matrix = A_prime_matrix - projection_onto_S_current @ A_prime_matrix
    
    # 2. 从 A' 中采样一个包含 10k 列的集合 C
    #    概率 p_i = ||E_{:i}||_F^2 / sum_j(||E_{:j}||_F^2)
    candidate_col_residual_norms_sq = np.zeros(d_prime)
    for i in range(d_prime):
        candidate_col_residual_norms_sq[i] = frobenius_norm_squared(E_ls_residual_matrix[:, [i]])
    
    sum_candidate_col_norms_sq = np.sum(candidate_col_residual_norms_sq)
    
    C_indices_for_swap_in = []
    if sum_candidate_col_norms_sq == 0:
        if d_prime > 0:
            print("警告: LS 中所有候选列的残差范数平方和为0。均匀采样C。")
            num_to_sample_C = min(10 * k_target_cols, d_prime)
            if num_to_sample_C > 0:
                 C_indices_for_swap_in = np.random.choice(d_prime, size=num_to_sample_C, replace=False)
        # else: C_indices 保持为空
    else:
        sampling_probs_for_C = candidate_col_residual_norms_sq / sum_candidate_col_norms_sq
        sampling_probs_for_C /= np.sum(sampling_probs_for_C) # 保证和为1
        num_to_sample_C = min(10 * k_target_cols, d_prime)
        if num_to_sample_C > 0 :
            try:
                C_indices_for_swap_in = np.random.choice(d_prime, size=num_to_sample_C, p=sampling_probs_for_C, replace=False)
            except ValueError as e: # 如果概率不合法（例如有负数或和不为1）
                print(f"警告: np.random.choice 采样C时出错: {e}. 将均匀采样。")
                if num_to_sample_C > 0 and d_prime >= num_to_sample_C :
                     C_indices_for_swap_in = np.random.choice(d_prime, size=num_to_sample_C, replace=False)


    if not C_indices_for_swap_in.size: # C 为空
        return S_current_col_indices_in_A_prime

    # 3. 从 C 中均匀采样一个索引 p (swap-in candidate)
    p_swap_in_col_idx = np.random.choice(C_indices_for_swap_in)

    # 5. 寻找最佳 q (swap-out candidate)
    best_q_swap_out_col_idx = -1
    q_list_position_to_replace = -1
    min_new_residual_sq_after_swap = current_total_residual_sq

    for i_q, q_current_col_idx in enumerate(S_current_col_indices_in_A_prime):
        if p_swap_in_col_idx == q_current_col_idx: # 换入和换出是同一列，则相当于不换，跳过
            continue

        # 构造临时的列索引集合，尝试用 p 替换 q
        temp_col_indices = list(S_current_col_indices_in_A_prime)
        temp_col_indices[i_q] = p_swap_in_col_idx 
        
        # 确保没有重复的列索引 (如果 p_swap_in_col_idx 已经是 S_current_col_indices_in_A_prime 中除了 q_current_col_idx 之外的某个列)
        if len(set(temp_col_indices)) < k_target_cols:
            continue 

        S_temp_matrix_from_A_prime = A_prime_matrix[:, temp_col_indices]
        
        # 检查秩，避免病态矩阵
        if S_temp_matrix_from_A_prime.shape[1] != k_target_cols or \
           np.linalg.matrix_rank(S_temp_matrix_from_A_prime) < k_target_cols:
            # print(f"警告: LS 中交换后矩阵秩亏。跳过此次交换尝试。索引: {temp_col_indices}")
            continue

        new_residual_sq = calculate_residual_error_squared(A_prime_matrix, S_temp_matrix_from_A_prime)

        if new_residual_sq < min_new_residual_sq_after_swap:
            min_new_residual_sq_after_swap = new_residual_sq
            best_q_swap_out_col_idx = q_current_col_idx
            q_list_position_to_replace = i_q # 记录 q 在 S_current_col_indices_in_A_prime 中的位置

    # 6 & 7. 如果找到了改进，则更新 S_current_col_indices_in_A_prime
    if best_q_swap_out_col_idx != -1:
        # print(f"  LS 交换: 换入列 {p_swap_in_col_idx}, 换出列 {best_q_swap_out_col_idx}. "+
        #       f"残差 {current_total_residual_sq:.4f} -> {min_new_residual_sq_after_swap:.4f}")
        updated_indices = list(S_current_col_indices_in_A_prime)
        updated_indices[q_list_position_to_replace] = p_swap_in_col_idx
        return sorted(updated_indices) 
    else:
        # print(f"  LS 未找到更优交换。当前残差: {current_total_residual_sq:.4f}")
        return S_current_col_indices_in_A_prime


def lscss_algorithm_1(A_original_matrix, k_cols_to_select, T_iterations_for_LS):
    """
    Algorithm 1: LSCSS (Linear time approximation algorithm for CSS with Local Search)
    A_original_matrix: 原始 n x d 矩阵
    k_cols_to_select: 要选择的列数 k
    T_iterations_for_LS: 局部搜索的迭代次数 T
    """
    n, d = A_original_matrix.shape
    
    if not (0 < k_cols_to_select <= d):
        raise ValueError("k_cols_to_select 必须在 (0, d] 范围内")

    print("--- LSCSS 算法开始 ---")
    print(f"步骤 1-11: 初始化和构造 A'")

    # --- 初始化阶段 (Algorithm 1, Steps 1-11) ---
    # B = A_original_matrix (Step 1)

    # t=1 循环 (Steps 3-6 for t=1): 在原始 A_original_matrix 上选择 k 列
    # 这些列用于计算 alpha。 E 的计算基于 A_original_matrix。
    print("  t=1: 在原始 A 上选择 k 列用于计算 alpha...")
    indices_for_alpha_calc = select_columns_for_initialization_or_LS(
        A_original_matrix, k_cols_to_select, A_original_matrix
    )
    if not indices_for_alpha_calc or len(indices_for_alpha_calc) < k_cols_to_select:
        print(f"警告: 为 alpha 计算选择的列数不足 {k_cols_to_select}。使用随机选择的列。")
        indices_for_alpha_calc = sorted(list(np.random.choice(d, k_cols_to_select, replace=False)))

    S_matrix_for_alpha = A_original_matrix[:, indices_for_alpha_calc]

    # 计算 alpha (Algorithm 1, Step 8, for t=1)
    # D_ii = alpha_value, where alpha_value = ||A - A_I A_I†A||_F / (denominator)^0.5
    residual_sq_for_alpha = calculate_residual_error_squared(A_original_matrix, S_matrix_for_alpha)
    
    try:
        # k_plus_1_factorial 可能非常大
        k_plus_1_fact = float(math.factorial(k_cols_to_select + 1))
    except OverflowError:
        k_plus_1_fact = float('inf')
        print(f"警告: (k+1)! 计算溢出，k={k_cols_to_select}。alpha 可能不准确。")


    denominator_for_alpha_calc = (52 * min(n, d) * k_plus_1_fact)
    if denominator_for_alpha_calc == 0 or denominator_for_alpha_calc == float('inf') or residual_sq_for_alpha == float('inf'):
        alpha_value = 1e-6 # 默认的小扰动值
        print(f"警告: 计算 alpha 的分母或分子有问题 (denom={denominator_for_alpha_calc}, resid_sq={residual_sq_for_alpha})。使用默认 alpha={alpha_value:.2e}。")
    else:
        alpha_value = np.sqrt(residual_sq_for_alpha) / np.sqrt(denominator_for_alpha_calc)
        if alpha_value == 0 or np.isnan(alpha_value) or np.isinf(alpha_value): 
            alpha_value = 1e-6 
            print(f"警告: 计算得到的 alpha 为0/nan/inf。使用默认 alpha={alpha_value:.2e}。")
    print(f"  计算得到 alpha = {alpha_value:.3e}")

    # 构造扰动矩阵 D 和 A_prime (Algorithm 1, Step 8 & 9, then effectively Step 12)
    # D 是 n x d 矩阵，D_ii = alpha_value
    D_perturbation_matrix = np.zeros_like(A_original_matrix)
    diag_length_for_D = min(n, d)
    for i in range(diag_length_for_D):
        D_perturbation_matrix[i, i] = alpha_value
    
    # A_prime = B + D (Step 12), B is original A (Step 1)
    A_prime_matrix = A_original_matrix + D_perturbation_matrix
    print(f"  已构造扰动矩阵 A' (形状: {A_prime_matrix.shape})")

    # t=2 循环 (Steps 3-6 for t=2): 在 A_prime_matrix 上选择 k 列
    # 这些列作为局部搜索的初始解。 E 的计算基于 A_prime_matrix。
    # 伪代码 Step 9 中 `set I=0`，所以是重新选择。
    print("  t=2: 在 A' 上选择 k 列作为局部搜索的初始解...")
    S_initial_indices_for_LS = select_columns_for_initialization_or_LS(
        A_prime_matrix, k_cols_to_select, A_prime_matrix
    )
    if not S_initial_indices_for_LS or len(S_initial_indices_for_LS) < k_cols_to_select:
        print(f"警告: 为局部搜索初始化选择的列数不足 {k_cols_to_select}。使用随机选择的列。")
        S_initial_indices_for_LS = sorted(list(np.random.choice(d, k_cols_to_select, replace=False)))
    
    current_S_indices_in_A_prime = S_initial_indices_for_LS
    print(f"  局部搜索的初始列索引 (在 A' 中): {current_S_indices_in_A_prime}")
    
    # --- 局部搜索阶段 (Algorithm 1, Steps 13-15) ---
    print(f"\n步骤 13-15: 执行局部搜索 {T_iterations_for_LS} 次...")
    for i_ls in range(T_iterations_for_LS):
        # print(f"  局部搜索迭代 {i_ls+1}/{T_iterations_for_LS}")
        current_S_indices_in_A_prime = local_search_step_LS(
            A_prime_matrix, k_cols_to_select, current_S_indices_in_A_prime
        )
    
    # --- 返回结果 (Algorithm 1, Steps 16-17) ---
    # Step 16: I 是局部搜索后在 A_prime 上的列索引
    # Step 17: return A_I (从原始 A_original_matrix 中提取列)
    final_selected_indices_in_A_original = current_S_indices_in_A_prime
    
    # 确保索引有效且数量正确 (理论上应该没问题)
    final_selected_indices_in_A_original = [idx for idx in final_selected_indices_in_A_original if 0 <= idx < d]
    final_selected_indices_in_A_original = sorted(list(set(final_selected_indices_in_A_original))) #去重并排序
    
    if len(final_selected_indices_in_A_original) < k_cols_to_select:
        print(f"警告: 最终选择的列数 {len(final_selected_indices_in_A_original)} 少于目标 k={k_cols_to_select}。尝试补充。")
        # ... (补充逻辑，如果需要，但理想情况下不应发生)
    elif len(final_selected_indices_in_A_original) > k_cols_to_select:
         print(f"警告: 最终选择的列数 {len(final_selected_indices_in_A_original)} 多于目标 k={k_cols_to_select}。截取前k个。")
         final_selected_indices_in_A_original = final_selected_indices_in_A_original[:k_cols_to_select]


    A_selected_submatrix = A_original_matrix[:, final_selected_indices_in_A_original]
    
    print("--- LSCSS 算法结束 ---")
    return A_selected_submatrix, final_selected_indices_in_A_original

# --- Demo ---
if __name__ == "__main__":
    # np.random.seed(42) 

    # 示例数据参数
    # n_rows, d_features, k_select = 20, 15, 6  
    # n_rows, d_features, k_select = 8, 6, 3      
    # n_rows, d_features, k_select = 208, 60, 5   # 稍大规模
    k_select = 5

    print(f"--- Demo 设置 ---")
    print(f"希望选择的列数 k: {k_select}")

    # A_orig = np.random.rand(n_rows, d_features)
    A_orig = pd.read_csv('datasets/sonar/sonar.csv', header=None).iloc[:, :-1].to_numpy()
    n_rows , d_features = A_orig.shape
    print(f"原始矩阵行数 n: {n_rows}")
    print(f"原始矩阵列数 d: {d_features}")
    print(A_orig.shape)

    # if k_select > 0 and k_select <= d_features:
    #     A_orig[:, :k_select] = A_orig[:, :k_select] * 3 + np.random.rand(n_rows, k_select) * 0.5 
    #     if k_select * 2 <= d_features:
    #          A_orig[:, k_select:k_select*2] = A_orig[:, :k_select] * 0.7 + np.random.rand(n_rows, n(k_select, d_features-k_select)) * 0.2


    if n_rows <= 10 and d_features <= 10:
        print(f"\n原始矩阵 A (形状 {A_orig.shape}):")
        print(A_orig)
    else:
        print(f"\n原始矩阵 A 形状: {A_orig.shape}")


    # 局部搜索迭代次数 T = O(k^2 log k)
    # if k_select <= 1:
    #     T_ls = 5 
    # else:
    #     # 使用math.log (自然对数), 论文中 log k 通常指 log_2 k 或自然对数，影响常数项
    #     T_ls = int(k_select**2 * math.log2(k_select) + 1) 
    #     T_ls = max(5, T_ls) # 保证至少迭代几次
    T_ls = 2 * k_select
    print(f"局部搜索迭代次数 T: {T_ls}\n")
    
    # 执行 LSCSS 算法
    selected_S_matrix, final_indices = lscss_algorithm_1(A_orig, k_select, T_ls)

    print(f"\n--- 结果 ---")
    final_indices_python_int = [int(idx) for idx in final_indices]
    print(f"LSCSS 选择的列索引: {final_indices_python_int}")
    if selected_S_matrix.shape[0] <= 10 and selected_S_matrix.shape[1] <= 10:
        print(f"选择的子矩阵 S (形状 {selected_S_matrix.shape}):")
        print(selected_S_matrix)
    else:
        print(f"选择的子矩阵 S 形状: {selected_S_matrix.shape}")


    # --- 性能评估 ---
    print(f"\n--- 性能评估 ---")
    # 理论最优误差（对于SVD截断, 作为参考下界）
    # 注意：CSS 的最优解通常不等于 SVD 低秩近似的投影误差。
    # ||A - A_k_svd||_F^2 是 rank-k 近似误差
    # ||A - P_k A||_F^2 是投影到最优k维子空间的误差，其中 P_k 是到A的前k个左奇异向量张成空间的投影
    # 对于CSS，我们寻找 ||A - S S_dagger A||_F^2
    U_svd, s_svd, Vt_svd = np.linalg.svd(A_orig, full_matrices=False)
    # 最优k维子空间投影误差 (由SVD的前k个左奇异向量张成的空间)
    A_projected_to_optimal_k_space = U_svd[:, :k_select] @ (U_svd[:, :k_select].T @ A_orig)
    optimal_k_subspace_residual_sq = frobenius_norm_squared(A_orig - A_projected_to_optimal_k_space)
    # 等价于 sum(s_svd[k_select:]**2)
    # optimal_k_subspace_residual_sq_direct = np.sum(s_svd[k_select:]**2)


    # 算法输出的残差
    lscss_residual_error_sq = calculate_residual_error_squared(A_orig, selected_S_matrix)

    print(f"  SVD 最优k维子空间投影残差平方 (||A - P_k A||_F^2, 参考下界): {optimal_k_subspace_residual_sq:.4f}")
    print(f"  LSCSS 选择的列构成的子矩阵 S 的残差平方 (||A - SS†A||_F^2): {lscss_residual_error_sq:.4f}")

    if optimal_k_subspace_residual_sq > 1e-9 : # 避免除以非常小或零的数
        error_ratio_vs_svd_space = lscss_residual_error_sq / optimal_k_subspace_residual_sq
        print(f"  误差比率 (LSCSS残差 / SVD最优k子空间残差): {error_ratio_vs_svd_space:.4f}")
    else:
        print(f"  SVD最优k子空间残差接近于0，无法计算有意义的比率。")

    # 对比：随机选择 k 列
    if d_features >= k_select:
        random_sel_indices = sorted(list(np.random.choice(d_features, k_select, replace=False)))
        A_random_sel_matrix = A_orig[:, random_sel_indices]
        random_sel_residual_sq = calculate_residual_error_squared(A_orig, A_random_sel_matrix)
        final_random_sel_indices = [int(idx) for idx in random_sel_indices]
        print(f"  随机选择 k 列的残差平方 (索引: {final_random_sel_indices}): {random_sel_residual_sq:.4f}")
        if optimal_k_subspace_residual_sq > 1e-9:
             random_error_ratio = random_sel_residual_sq / optimal_k_subspace_residual_sq
             print(f"    误差比率 (随机残差 / SVD最优k子空间残差): {random_error_ratio:.4f}")

    # 对比：选择范数最大的 k 列
    if d_features >= k_select:
        col_frobenius_norms = np.linalg.norm(A_orig, axis=0)
        top_norm_sel_indices = sorted(list(np.argsort(col_frobenius_norms)[-k_select:]))
        A_top_norm_sel_matrix = A_orig[:, top_norm_sel_indices]
        top_norm_sel_residual_sq = calculate_residual_error_squared(A_orig, A_top_norm_sel_matrix)
        final_top_norm_sel_indices = [int(idx) for idx in top_norm_sel_indices]
        print(f"  选择范数最大的 k 列的残差平方 (索引: {final_top_norm_sel_indices}): {top_norm_sel_residual_sq:.4f}")
        if optimal_k_subspace_residual_sq > 1e-9:
            top_norm_error_ratio = top_norm_sel_residual_sq / optimal_k_subspace_residual_sq
            print(f"    误差比率 (TopNorm残差 / SVD最优k子空间残差): {top_norm_error_ratio:.4f}")