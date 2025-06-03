你好，我正在进行一个算法课程project的学习，我们选择了CSS问题来研究。以下是一篇最新工作的部分介绍，但由于缺乏基础，很多地方我不能明白，且不能把其中的定义、定理和算法联系起来，理解它们的作用。能否请你为我细致入微、不厌其烦地讲解一下呢。讲解可以使用中英文结合，既方便理解，又不贸然改变原意

#### Definition 1.1.

 Given a matrix $A \in \mathbb{R}^{n \times d}$ and a positive integer $k$, the goal of CSS problem is to select $k$ columns of $A$ forming a matrix $S \in \mathbb{R}^{n \times k}$ that minimizes the residual error $\|A - SS^{\dagger}A\|_{F}^{2}$, where $S^{\dagger}$ represents the Moore-Penrose inverse matrix of $S$, and $\|A\|_{F}^{2} = \sum_{i=1}^{n} \sum_{j=1}^{d} A_{ij}^{2}$ denotes the square of Frobenius norm.

### Preliminaries

For any positive integer $n$, let $[n]$ denote the set ${1,2, \dots, n}$. Given a matrix $A \in \mathbb{R}^{n \times d}$, let $A_{ij}$ be the element in the $i$-th row and the $j$-th column of $A$, and define the Frobenius norm of $A$ as $\|A\|_{F}^{2}=\sum_{i=1}^{n}\sum_{j=1}^{d}A_{ij}^{2}$. Denote $A_{:j}$ as the $j$-th column of $A$, and $A_{:i}$: as the $i$-th row of $A$. Let $A^{\top}$ be the transpose of $A$ and $A^\dagger$ be the Moore-Penrose inverse of $A$. Given an $n \times d$ matrix $A$, let $\mathcal{I}$ be the set of column indices from A, and let $A_{\mathcal{I}}$ denote the $n \times |\mathcal{I}$ submatrix of A consisting of the columns corresponding to the indices in $\mathcal{I}$. For a matrix $A$, the linear span of its column vectors is denoted as $span(A)$. For any two $n \times d$ matrices $A$ and $B$, $\|AB\|_F \leq \|A\|_F\|B\|_2$ and $\|AB\|_F \leq \|A\|_2\|B\|_F$. Given any matrix $A \in \mathbb{R}^{n \times d}$, the singular value decomposition (SVD) of $A$ can be written as $A=\sum_{i=1}^{\min(n,d)}\sigma_{i}u_{i}v_{i}^{\top}$, where $\sigma_1 \geq \dots \sigma_n \geq 0$ are the singular values, ${u_1, \dots, u_n} \subseteq \mathbb{R}^n$ are the left singular vectors, and ${v_1, \dots, v_d} \subseteq \mathbb{R}^d$ are the right singular vectors. Denote $rank(A)$ be the rank of a matrix $A$, which is the number of non-zero singular values of $A$. Moreover, we denote $A_k = \sum_{i=1}^{k}\sigma_{i}u_{i}v_{i}^{\top}$ as the best rank-$k$ approximation to A under the Frobenius norm. The spectral norm of $A$, denoted by $\|A\|_2$, is defined as the largest singular value of $A$, i.e., $\|A\|_2 = \sigma_{max}(A)$. Given a solution $S$ to the CSS problem on matrix $A$, we define the residual error of $S$ as $f(A,S) = \|A-SS^{\dagger}A\|_{F}^{2}$.

### LinearTime Local Search Algorithm for CSS Problem

In this section, we propose a local search approximation algorithm for solving the CSS problem, called LSCSS, which maintains a running time linear in both n and d. Directly applying single-swap local search to solve the CSS problem results in an $O(nd^2k^2)$ running time by enumerating all possible swap pairs. Thus, it is challenging to apply the local search method to solve the CSS problem while maintaining a linear dependence on both n and d in the running time. To avoid O(nd2k2) running time in each local search step, we propose a two-step mixed sampling method to reduce the running time from $O(nd^2k^2)$ to $O(ndk^2)$. Although the sampling method reduces the running
 time by directly using the single-swap local search, analyzing the bound of improvement on the residual error after swaps is a difficult task. To provide a theoretical analysis for the local search step, we propose a matched swap pair construction method to bound the improvement on the residual error during swaps. By carefully analyzing the improvement, we show that our proposed algorithm
 achieves $53(k+1)$-approximation with $O(ndk^4 \log k)$ running time. The detailed algorithm for the CSS problem is given in Algorithm 1.

The LSCSS algorithm mainly comprises local search and two-stage mixed sampling components. The high-level idea behind our proposed local search is to identify a swap pair that minimizes the residual error in each iteration. The swap pair consists of a column from the input matrix to swap in and a column from the current solution to swap out. By repeating this process, the algorithm produces an updated solution with better quality. Moreover, the two-stage mixed sampling method involves two steps for obtaining a candidate column from the input matrix. Firstly, a set of column indices is constructed by sampling each column with probability proportional to its residual error for the current solution. Then, a column is uniformly selected from the set of candidate indices as the final column to swap in. To ensure that the input matrix for the local search process is full rank, we construct a new matrix $A'$ by adding a small perturbation matrix D to the original matrix $A$ during the initialization, where $D$ is full-rank and has non-zero values only on its diagonal. The full-rank property of $A'$ is used in subsequent analysis.

The LSCSS algorithm begins by obtaining an initial solution S with exactly k columns and constructing a full-rank matrix $A'$ during the initialization (steps 1-12 of Algorithm 1), which achieves a $(k+1)!$-approximate solution on $A'$. We start by initializing an index set $\mathcal{I}$ and setting the matrix $E=A$. A new column index is added to $\mathcal{I}$ by sampling each column indexii from $[d]$ with probability proportional to $p_{i}=\|E_{:i}\|_{2}^{2}/\|E\|_{F}^{2}$. Then, $E$ is updated as $E=A-A_{\mathcal{I}}A_{\mathcal{I}}^{\dagger}A$. Repeating this process $k$ times, we obtain an initial solution $S=A_{\mathcal{I}}$. To construct a full-rank matrix, we construct an $n \times d$ zero matrix $D$ and compute the parameter $\alpha=\|A-SS^{\dagger}A\|_{F}/(52\operatorname*{min}\{n,d\}(k+1)!)^{-1/2}$ using the initial solution $S$ and $A$. Each diagonal entry $D_{ii}$ is set to $\alpha$. Since $rank(A + D) = rank(D)$, we construct the full-rank matrix $A'$ by adding the full-rank matrix $D$ to the input matrix $A$. To solve the CSS problem on $A'$, we execute steps 3-6 of Algorithm 1 to obtain the solution $S=A_{\mathcal{I}}'$. The detailed process described in steps 1-12 of Algorithm 1 requires $O(ndk^2)$ time. The local search performed in steps 13-15 of Algorithm 1 plays a crucial role in LSCSS, involving two main steps. Firstly, we compute the matrix $E=A-SS^{\dagger}A$ for the current solution $S$. Then, a set $C$ of 10$k$ column indices is constructed by sampling each column index $i$ from $[d]$ with probability $p_{i}=\|E_{:i}\|_{2}^{2}/\|E\|_{F}^{2}$. Next, a column index $p$ is uniformly selected from $C$, referred to as the “swapin” column index. Let $\mathcal{I}$ denote the set of column indices of $S$ in $A'$. Subsequently, if there exists an index $q \in \mathcal{I}$ such that $f(A^{\prime},A_{\mathcal{I}\setminus\{q\}\cup\{p\}}^{\prime})<f(A^{\prime},S)$, we choose $q$ as the “swap-out” column index and update the set of indices to $\mathcal{I} = \mathcal{I} \setminus\{q\} \cup\{p\}$. Finally, Algorithm 2 returns the solution $S=A_{\mathcal{I}}'$. After repeating this process $T=O(k^2 \log k)$ times, Algorithm 1 returns the final solution $S$ for the input matrix $A$.

#### Algorithm 1 LSCSS

- Input: a matrix $A \in \mathbb{R}^{n \times d}$, an integer $k$, and the number of iterations $T$
- Output: a submatrix consisting of $k$ columns from $A$
- 1: Initialize $\mathcal{I} = \emptyset$, $E = A$, $B = A$.
- 2: for $t = 1, 2$ do
- 3:     for $j \leftarrow 1, 2, \ldots, k$ do
- 4:         Sample a column index $i \in [d]$ with probability $p_i = \|E_{:,i}\|_2^2/\|E\|_F^2$.
- 5:         Update $\mathcal{I} = \mathcal{I} \cup \{i\}$ and $E = A - A_{\mathcal{I}} A_{\mathcal{I}}^\dagger A$.
- 6:     end for
- 7:     if $t$ is equal to 1 then
- 8:         Initialize an $n \times d$ zero matrix $D$, and set each diagonal entry $D_{ii} = \frac{\|A - A_{\mathcal{I}} A_{\mathcal{I}}^\dagger A\|_F}{(52 \min\{n, d\}(k+1)!)^{1/2}}$.
- 9:         Compute $A \leftarrow A + D$ and set $\mathcal{I} = \emptyset$.
- 10:     end if
- 11: end for
- 12: Compute $A' = B+D$ and set $S=A'_{\mathcal{I}}$.
- 13: for $i \leftarrow 1, 2, \ldots, T$ do
- 14:     $S \leftarrow \text{LS}(A', k, S)$
- 15: endfor
- 16: Let $\mathcal{I}$ be the set of column indices of $S$
- 17: return $A_{\mathcal{I}}$

#### Algorithm 2 LS

- Input: a matrix $A' \in \mathbb{R}^{n \times d}$, an integer $k$, and a matrix $S \in \mathbb{R}^{n \times k}$
- Output: a submatrix consisting of $k$ columns from $A'$
- 1: Compute the residual matrix $E = A' - SS^\dagger A'$.
- 2: Sample a set $C$ of $10k$ column indices from $A'$, where each column index $i$ is picked with probability $\|E_{:,i}\|_2^2/\|E\|_F^2$.
- 3: Uniformly sample an index $p \in C$.
- 4: Let $\mathcal{I}$ be the set of the columns indices of $S$ in $A'$.
- 5: if there exists an index $q \in \mathcal{I}$ such that $f(A', A'_{\mathcal{I}\setminus\{q\} \cup \{p\}}) < f(A', S)$ then
- 6:     Find an index $q \in \mathcal{I}$ that minimizes $f(A', A'_{\mathcal{I}\setminus\{q\} \cup \{p\}})$.
- 7:     $\mathcal{I} = \mathcal{I} \setminus \{q\} \cup \{p\}$.
- 8: end if
- 9: return $A'_{\mathcal{I}}$.

#### Analysis

In the following, we explain in more detail how our proposed local search algorithm achieves a $53(k+1)$-approximation for the original matrix $A$. Given an initial solution, the main idea for analyzing the approximation ratio of our algorithm is to bound the improvement on residual error during the swaps in the local search step. To achieve this bound, we propose a matched swap pair construction method that guarantees an improvement in the current solution by swapping one column (Lemma 3.6 and Lemma 3.7. By carefully analyzing the improvement, we show that with constant probability the approximation loss of the current solution can be reduced by a multiplicative factor of $1-\Theta(1/k)$ in each iteration of the local search algorithm (Lemma 3.8. This implies that after $O(k^{2}\log k)$ iterations, we have $\|A^\prime-SS^\dagger A^{\prime}\|_{F}^2\leq 26(k+1)\|A^{\prime}-A_k^{\prime}\|_{F}^2$ (Theorem 3.9. Finally, by analyzing the change in the residual error caused by removing matrix $D$ from $A^{\prime}$,we obtain $\|A^{\prime}-SS^{\dagger}A^{\prime}\|_{F}^{2}\leq 53(k+1)\|A-A_{k}\|_{F}^{2}$ in expectation (Lemma 3.10).
We assume that the matrix $A$ has been normalized such that $\|A\|_F^2=1/4$. Otherwise, we can normalize each element $A_{ij}$ in $A$ as $A_{ij}=\frac{A_{ij}}{2\|A\|_F}.$ Next, we consider a single iteration of Algorithm 2. We assume that the current solution has a high residual error (larger than $25(k+1)\|A^\prime-A_k^\prime\|_F^2$ before executing Algorithm 2 on $A^\prime$. Otherwise, the initial solution $S$ is a $25(k+1)$-approximation for the input matrix $A^{\prime}$.

Let $S^{*} = \{s_{1}^{*},\dots,s_{k}^{*}\}$ be the optimal solution with exactly $k$ selected for $A'$, and let $S = \{s_{1},\dots,s_{k}\}$ be the current solution. We define $\phi(A^{\prime},S^{*},S,s^{*}) = \arg\min_{s\in S}f(A^{\prime},S^{*}\setminus\{s^{*}\}\cup\{s\})$ as a mapping function that finds $s$ from $S$ such that the residual error $f(A^{\prime},S^{*}\setminus\{s^{*}\}\cup\{s\})$ is minimized.Thus, we say that $s^*$ is captured by $\phi(A^{\prime},S^*,S,s^*)$. Each column $s^*\in S^*$ is captured by exactly one column from $S$. Let$\mathcal{I}$ denote the set of column indices of $S$ in matrix $A^\prime$. We denote $\mathcal{L}$ as the set of columns indices in $S$ that do not capture any optimal columns. We denote $\mathcal{H}$ as the set of indices where each column in $S$ captures exactly one optimal column.

The main idea behind the matched swap pair construction method is to analyze the change in residual error caused by swapping an index from set $\mathcal{H}$ (or $\mathcal{L}$)with an index from a sampled column, using a two-step mixed sampling approach for the current solution $S$. For the column $s_h$ (where $h\in \mathcal{H}$) in $S, $s_h$ captures exactly the column $s_h^*$ of the optimal solution $S^*$,serving as the candidate column for $s_h^*$. If the residual error of swapping $s_h$ to replace $s_h^*$ is large, we prove that with constant probability, sampling a new column can reduce the residual error and update $s_h$. Similarly, for the column $s_l$ (where $l\in \mathcal{L}$) in $S,s_l$ does not match any optimal column. We also show that, with constant probability, sampling a column from the input matrix $A^{\prime}$ can reduce the residual error for columns in set $\mathcal{L}$ To analyze the improvement in residual error during swaps, we focus on a single swap process, evaluating both the increase in residual error from removing a column $s$ from $S$ and the decrease in residual error from inserting a new column. We give the following definition to measure the change resulting from removing a column.

#### Definition 3.1.
Let $A' \in \mathbb{R}^{n \times d}$ be a full-rank matrix, and let $S$ be a solution on $A'$. Let $\mathcal{I}$ be the set of column indices of $S$. The change in residual error by removing the column $i$ from $\mathcal{I}$ is defined as 
$$\tau(A', S, \mathcal{I} \setminus \{i\}) = f(A', A'_{\mathcal{I} \setminus \{i\}}) - f(A', S).$$
To bound $\tau(A', S, \mathcal{I} \setminus \{i\})$ of solution $S$ on the matrix $A'$, we provide the theoretical guarantee in the following lemma. (Detailed proof of Lemma 3.2 is given in Appendix A.1)

#### Lemma 3.2.
Let $A' \in \mathbb{R}^{n \times d}$ be a full-rank matrix, and let $S$ be a solution on $A'$. Let $\mathcal{I}$ be the set of the column indices in $S$. For $i \in \mathcal{I}$, we have
$$
\tau(A', S, \mathcal{I} \setminus \{i\}) \leq \|A'_{\mathcal{I}} {A'}_{\mathcal{I}}^{\dagger} A'\|_F^2
$$
To further analyze the bound on $\tau(A', S, \mathcal{I} \setminus \{i\})$, we decompose the projection matrix $A'_{\mathcal{I}} {A'}_{\mathcal{I}}^\dagger A'$ and show that the expected upper bound of $\tau(A', S, \mathcal{I} \setminus \{i\})$ is proportional to $\|A'\|_F^2$. (Detailed proof of Lemma 3.3 is given in Appendix A.1)

#### Lemma 3.3.
Let $A' \in \mathbb{R}^{n \times d}$ be a full-rank matrix, $k$ be a positive integer, and let $\mathcal{I}$ be the set of column indices of $S$ for the CSS problem on $A'$. In expectation, the following inequality holds
$$\|A'_{\mathcal{I}} {A'}_{\mathcal{I}}^\dagger A'\|_F^2 \leq \frac{k^2}{d^2} \|A'\|_F^2.$$
In the following, we theoretically bound the residual error resulting from adding a candidate column index $p$ to the set $\mathcal{I}$ of column indices in $S$, where $p$ is chosen using the two-step mixed sampling method. (Detailed proof of Lemma 3.4 is given in Appendix A.1)

#### Lemma 3.4.
Let $A' \in \mathbb{R}^{n \times d}$ be a full-rank matrix, $k$ be a positive integer, and let $S$ be a solution with the set $\mathcal{I}$ of column indices in $S$. Let $E = A' - S S^\dagger A'$. The column index $p$ is obtained by executing steps 2-3 of Algorithm 2. In expectation, the following inequality holds
$$
f(A', A'_{\mathcal{I} \cup \{p\}}) \leq f_k(A', \text{opt}) + \frac{1}{10} f(A', S),$$
where $f_k(A', \text{opt})$ denotes the best rank-$k$ solution.

According to the aforementioned mapping function $\phi(\cdot)$, we obtain the subset $H$ from the set $\mathcal{I}$ of column indices and the set $R=\mathcal{I}\setminus \mathcal{H}.$ By using the matched swap pair construction method, there are two cases for the residual error of the current solution:

1. For the set $H$,where $\sum_{h\in H}f(A^{\prime},A_{\mathcal{I}\setminus\{h\}}^{\prime})>\frac{21}{50}\sum_{i\in\mathcal{I}}f(A^{\prime},A_{\mathcal{I}\setminus\{i\}}^{\prime}).$
2. For the set $R=\mathcal{I}\backslash H$, where $\sum_{r\in R}f(A^{\prime},A^{\prime}_{\mathcal{I}\setminus\{r\}})\geq\frac{29}{50}\sum_{i\in\mathcal{I}}f(A^{\prime},A_{\mathcal{I}\setminus\{i\}}^{\prime}).$

By Lemma 3.2 and Lemma 3.4 we define the good columns $s_i$ for $i\in\mathcal{I}$ with respect to $S$ as follows.

#### Definition 3.5.
$LetA^\prime\in\mathbb{R}^{n\times d}$ be a full-rank matrix, and let k be a positive integer. Let S' be the optimal solution with exactly k columns selected, and let $T^*betheset\textit{ofcolumn indices in S}^*.$ Let $\bar{S}$ be any solution with exactly k columns selected, and let I be the set of column indices in $S.$ A columnindex $i\in \mathcal{I}$ $is$ called good $if$
$$\begin{aligned}&f(A^{\prime},A_{\mathcal{I}\setminus\{i\}}^{\prime})-\tau(A^{\prime},S,\mathcal{I}\setminus\{i\})-\tau(A^{\prime},A_{\mathcal{I}\cup\{p\}}^{\prime},(\mathcal{I}\cup\{p\})\backslash\{i\})\\&-\frac{11}{10}\left(f(A^{\prime},A^{\prime}_{\mathcal{I}^{*}\setminus\{i^{*}\}})+\frac{1}{10}f(A^{\prime},S)\right)>\frac{1}{100k}f(A^{\prime},S),\end{aligned}$$
where $i^* \in {\mathcal{I} }^*$ $is$ the column index mapped from $i\in {\mathcal{I} }$ $by$ the function $\phi ( \cdot ) , and$ $p$ $is$ $a$ column index obtained by executing steps 2-3 of Algorithm 2.

Definition 3.5 estimates the gain from replacing the column $s_h$ with a new column obtained using the two-step sampling method. Next, we argue that if case (1) happens, the sum of residual errors for the good columns is large. (Detailed proof of Lemma 3.6 is given in Appendix A.1)

#### Lemma 3.6.
Let $A^{\prime}\in\mathbb{R}^{n\times d}$ be a full-rank matrix, k be a positive integer, and let $S$ be the solution to the CSS problem on $A^{\prime }.$ Let $I$ be the set of column indices in $S.$ If $50\sum _{h\in H}f( A^{\prime }, A_{\mathcal{I} \setminus \{ h\} }^{\prime }) \geq$ $21\sum _{i\in \mathcal{I} }f( A^{\prime }, A_{\mathcal{I} \setminus \{ i\} }^{\prime })$ and $f( A^{\prime }, S) \geq 25( k+ 1) f_{k}( A^{\prime }, opt)$, we have
$$\sum_{h\in H,h\:is\:good}f(A',A'_{\mathcal{I}\setminus\{h\}})\ge\frac{1}{125}\sum_{i\in\mathcal{I}}f(A',A'_{\mathcal{I}\setminus\{i\}}).$$


Since $R=\mathcal{I}\backslash H$, it holds that $L\subseteq R.$ The index set $R$ contains two subsets: $L$ and $R\backslash L$, where the indices in $L$ do not capture any optimal columns according to the mapping function $\phi(\cdot)$ and the indices in $R\backslash L$ capture at least two columns. Similar to case (1), we argue that if case (2) occurs, the sum of residual errors for the good columns is large. (Detailed proof of Lemma 3.7 is given in Appendix A.1)

#### Lemma 3.7.
Let $A' \in \mathbb{R}^{n \times d}$ be a full-rank matrix, $k$ be a positive integer, and let $S$ be a solution for the CSS problem on matrix $A'$. Let $\mathcal{I}$ be the set of column indices in $S$. If $\sum_{r \in R} f(A', A'_{\mathcal{I} \setminus \{r\}}) \geq \frac{29}{50} \sum_{i \in \mathcal{I}} f(A', A'_{\mathcal{I} \setminus \{i\}})$ and $f(A', S) \geq 25(k + 1)f_k(A', opt)$, we have
$$\sum_{r \in R, r \text{ is good}} f(A', A'_{\mathcal{I} \setminus \{r\}}) \geq \frac{1}{125} \sum_{i \in \mathcal{I}} f(A', A'_{\mathcal{I} \setminus \{i\}}).$$
In the following, we prove that if the residual error of the current solution is larger than $25(k + 1)f_k(A', opt)$, each local search step reduces the residual error by a factor of $1 - \Theta(\frac{1}{k})$ with constant probability. (Detailed proof of Lemma 3.8 is given in Appendix A.1)

#### Lemma 3.8.
Let $A' \in \mathbb{R}^{n \times d}$ and $S \in \mathbb{R}^{k \times d}$ be the input matrices for Algorithm 2 where $k$ is a positive integer and $S$ is the solution of the CSS problem on $A'$. Suppose that $f(A', S) \geq 25(k + 1) \cdot f_k(A', opt)$. Then, with probability at least $1/1375$, Algorithm 2 returns a new solution $S'$ with
$$
f(A', S') \leq (1 - 1/(100k))f(A', S).$$
Subsequently, we prove that the LSCSS algorithm achieves a $26(k + 1)$-approximation for $A'$ after $O(k^2 \log k)$ iterations.
Theorem 3.9. Let $A' \in \mathbb{R}^{n \times d}$ be the input matrix obtained in step 12 of Algorithm 1, let $k$ be a positive integer, and let $S$ be the solution returned after executing Algorithm 2 $T = O(k^2 \log k)$ times. Then, it holds that
$$\mathbb{E}[\|A' - SS^\dagger A'\|_F^2] \leq 26(k + 1)\|A' - A'_k\|_F^2,$$

where $A'_k$ is the best rank-$k$ approximation of $A'$ for the CSS problem. The running time of Algorithm 1 is $O(ndk^4 \log k)$.

**Proof.** Let $\hat{S}$ denote the submatrix consisting of $k$ columns obtained in step 12 of Algorithm 1. For the initial solution $\hat{S}$, Deshpande and Vempala [15] provide an approximation ratio of $(k+1)!$. Before executing steps 13-15 of Algorithm 1, the residual error of the initial solution $\hat{S}$ is larger than $25(k+1)\|A'-A'_k\|_F^2$. According to Lemma 3.8, with probability $1/1375$, we can reduce the residual error by a multiplicative factor of $(1-1/100k)$.
Let $T=O(k^2\log k)$. We define a random process $\mathcal{P}$ with initial residual error $\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2$ of the solution $\hat{S}$ such that for $T$ iterations of Algorithm 2, it reduces the value of $\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2$ by at least $(1-1/100k)$ with probability $1/1375$, and it increases the final value of $\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2$ by $25(k+1)\|A'-A'_k\|_F^2$. It is obvious that $\mathbb{E}[\|A'-SS^\dagger A'\|_F^2]\leq\mathbb{E}[\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2]$. Then, we have
$$
\begin{aligned} \mathbb{E}[\mathcal{P}]
&=25(k+1)\|A'-A'_k\|_F^2+\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2\cdot\sum_{i=0}^{T}\binom{T}{i}\frac{i}{1375}\frac{1374^{T-i}}{1375}\left(1-\frac{1}{100k}\right)^i \\
&\leq\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2\cdot\left(1-\frac{1}{137500k}\right)^{137500k\log(k+1)!}+25(k+1)\|A'-A'_k\|_F^2 \\
& \leq\frac{\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2}{(k+1)!}+25(k+1)\|A'-A'_k\|_F^2.\end{aligned}
$$
This implies that $\mathbb{E}[\|A'-SS^\dagger A'\|_F^2|\hat{S}]\leq\frac{\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2}{(k+1)!}+25(k+1)\|A'-A'_k\|_F^2$.

Thus, we obtain
$$
\begin{aligned}
\mathbb{E}[\|A'-SS^\dagger A'\|_F^2]
&=\sum_{\hat{S}}\mathbb{E}[\|A'-SS^\dagger A'\|_F^2|\hat{S}]Pr(\hat{S}) \\
&\leq\sum_{\hat{S}}Pr(\hat{S})\left(\frac{\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2}{(k+1)!}+25(k+1)\|A'-A'_k\|_F^2\right) \\
&\leq\frac{E[\|A'-\hat{S}\hat{S}^\dagger A'\|_F^2]}{(k+1)!}+25(k+1)\|A'-A'_k\|_F^2.
\end{aligned}
$$

Since $||A' - SS^{\dagger}A'||_F^2 \leq (k + 1)!\|A' - A'_k\|_F^2$ in expectation, we have $\mathbb{E}[||A' - SS^{\dagger}A'||_F^2] \leq 26(k + 1)\|A' - A'_k\|_F^2$.
#### Running Time Analysis.
In LSCSS algorithm, the process of constructing the initial solution in the steps 2-11 of Algorithm 1 takes $O(ndk^5)$ time. In order to obtain an $O(k + 1)$-approximate solution, Algorithm 2 requires $O(k^2 \log k)$ iterations. In each iteration, computing the residual matrix requires $O(ndk)$ time. The steps 4-8 of Algorithm 2 require $O(ndk^2)$ time to recalculate the residual error. Therefore, the overall running time of Algorithm 1 is $O(ndk^4 \log k)$.

In the following, we analyze the change in residual error caused by replacing the input matrix $A$ with $A' = A + D$, which leads to the final solution of Algorithm 1 achieving a $53(k + 1)$-approximation. (Detailed proof of Lemma 3.10 is given in Appendix A.1)

#### Lemma 3.10.
Let $A \in \mathbb{R}^{n \times d}$ be an input matrix, and let $k$ be a positive integer. Define $D$ as an $n \times d$ matrix with elements
$$D_{ij} = \begin{cases}
\frac{\|A - S_1S_1^{\dagger}A\|_F}{(52 \min\{n, d\}(k + 1)!)^{1/2}}, & \text{if } i = j \\
0, & \text{otherwise}
\end{cases},$$
where $S_1$ is obtained by executing the first round of steps 3-6 in Algorithm 1. Let $A' = A + D$. The solution $S_2$ returned by executing Algorithm 2 for $T = O(k^2 \log k)$ iterations satisfies
$$\mathbb{E}[||A' - S_2S_2^{\dagger}A'||_F^2] \leq 53(k + 1)\|A - A_k\|_F^2,$$

where $A_k$ is the best rank-$k$ approximation for $A$.

### Experiments
In this section, we compare our algorithm for the CSS problem with the previous ones. For hardware, all the experiments are conducted on a machine with 72 Intel Xeon Gold 6230 CPUs and 2TB memory.

Datasets. In this paper, we evaluate the performance of our algorithms on a total of 22 real-world datasets. In previous studies [23,1], the CSS problem typically involves datasets with no more than 100,000 rows and 20,000 columns. We include the 14 smaller datasets listed in Table [5](Appendix A.2). To extend the evaluation to larger datasets, we include 8 additional datasets detailed in Table [2]. Six datasets contain between 40,000 and 480,000 columns, and two contain 400,000 and 8 million rows, respectively. All datasets can be found on the website [35].

Algorithms and parameters. In our experimental evaluation, we consider the following five distinct algorithms:

- TwoStage. This is a two-stage algorithm from [5] that combines leverage score sampling and rank-revealing QR factorization.
- Greedy. This is an algorithm in [16,1], which uses greedy algorithm to generate solution.
- VolumeSampling. This is an algorithm in [13], which uses volume sampling method.
- ILS. This is an algorithm in [23], which uses heuristic local search method.
- LSCSS. This is our algorithm given in Algorithm [1], which uses the two-step mixed sampling and local search methods.

Methodology We use the error ratio to evaluate the effectiveness of various algorithms, as defined in [23]. The error ratio is given by the formula $||A - SS^{\dagger}A||_F^2 / ||A - A_k||_F^2$, where it quantifies the discrepancy between the selected columns and the optimal rank-$k$ matrix approximation. A smaller error ratio indicates better algorithm performance. Following [23], we test the TwoStage, VolumeSampling, ILS, and LSCSS algorithms on each dataset 10 times to calculate the average error ratio and running time. Since the Greedy algorithm is deterministic, it is tested only once per dataset.

Experimental setup. For the CSS problem with the Frobenius norm, we run the TwoStage, Greedy, ILS, and LSCSS algorithms on both the 8 large datasets and 14 small datasets, providing the average results for each method. The ILS and our LSCSS algorithm are based on local search method. For fair comparison, we set the number of iterations to be $2\breve{k}$ for ILS and LSCSS. Since the VolumeSampling requires $O(dkn^{3}\log n)$ runtime and $O(n^{2}+d^{2})$ memory, it cannot handle the 8 large datasets because the algorithm requires more than 48 hours of runtime and over 2TB of memory. However, the other four algorithms generally produce a solution within 48 hours and with less than 2TB of memory. Thus, we only include VolumeSampling in the comparison on the 14 smaller datasets and exclude its results from Tables 3 and 4.