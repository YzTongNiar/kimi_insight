# Kimi K3 Speculative Decode Analysis

本文记录 Kimi K3/KDA 在 speculative decoding 中的状态管理。下面的状态重建流程依据
[vllm-ascend PR #19087](https://github.com/vllm-project/vllm-ascend/pull/19087)
中的 `kda_recoverssm_verify` 和 `kda_recoverssm_commit` 实现整理。

## 数学角度的 KDA State 重建流程

### 1. KDA 单步递推

设一个 KDA head 的 recurrent state 为

$$
S_t \in \mathbb{R}^{V\times K},
$$

其中 $K$ 是 key/query 维度，$V$ 是 value 维度。第 $t$ 个 token 的输入包括
$q_t,k_t,v_t$、原始 gate $g^{\rm raw}_t$ 和原始 beta $\beta^{\rm raw}_t$。PR
中的实现先对 Q/K 做不带 $10^{-6}$ 偏置的 L2 归一化：

$$
\hat k_t=
\begin{cases}
k_t/\lVert k_t\rVert_2,&\lVert k_t\rVert_2>0,\\
0,&\text{otherwise},
\end{cases}
\qquad
\hat q_t=\frac{1}{\sqrt K}
\begin{cases}
q_t/\lVert q_t\rVert_2,&\lVert q_t\rVert_2>0,\\
0,&\text{otherwise}.
\end{cases}
$$

令 $a=\exp(A_{\log})$，$b$ 为 `dt_bias`。gate 的两种形式为：

$$
\gamma_t=
\begin{cases}
-a\,\operatorname{softplus}(g^{\rm raw}_t+b),&\text{无 lower bound},\\
\ell\,\sigma\!\left(a(g^{\rm raw}_t+b)\right),&\text{有 lower bound }\ell.
\end{cases}
\qquad d_t=\exp(\gamma_t).
$$

这里的 $d_t\in\mathbb{R}^{K}$ 是沿 key 维度逐列施加的衰减因子。先衰减旧状态：

$$
\widetilde S_t=S_{t-1}\operatorname{diag}(d_t).
$$

随后用 beta 对 value 误差做 rank-1 修正：

$$
u_t=\sigma(\beta^{\rm raw}_t)\odot
\left(v_t-\widetilde S_t\hat k_t\right),
\qquad
S_t=\widetilde S_t+u_t\hat k_t^{\mathsf T}.
$$

最终输出为

$$
y_t=S_t\hat q_t.
$$

因此，单步状态变换是一个“按列衰减 + rank-1 加法”的仿射变换。

### 2. verify 阶段：从 checkpoint 展开候选链

一次 speculative request 从已提交的 checkpoint $S_0$ 开始，沿候选 token 链前向展开
$L$ 个 verified positions：

$$
S_0\xrightarrow{(k_1,g_1,u_1)}S_1
\xrightarrow{(k_2,g_2,u_2)}\cdots
\xrightarrow{(k_L,g_L,u_L)}S_L.
$$

`kda_recoverssm_verify` 对每个位置执行上面的递推并生成 $y_1,\ldots,y_L$，但**不改写
checkpoint**。为了让后续 commit 能重建状态，它只保存每个位置的紧凑记录：

$$
R_t=(u_t,k_t,g^{\rm raw}_t),
$$

其中 correction 向量 $u_t$ 的大小为 $V$，原始 key/raw-gate 记录的大小为 $2K$。commit 会对记录中的 key 再做同样的归一化。因此每个
候选位置的记录开销是 $O(V+2K)$，而不是保存一个新的 $V\times K$ 矩阵。注意，保存的
gate 是 raw gate；commit 会用同样的 $A_{\log}$、`dt_bias` 和 lower bound 重新计算
$d_t$，从而保持与 verify 的数值路径一致。

### 3. 接受前缀的闭式状态表达

令 $D_t=\operatorname{diag}(d_t)$。由
$S_t=S_{t-1}D_t+u_t\hat k_t^{\mathsf T}$ 递推可得，接受前缀长度为 $n$ 时：

$$
S_n
=S_0D_1D_2\cdots D_n
+\sum_{i=1}^{n}
u_i\hat k_i^{\mathsf T}D_{i+1}D_{i+2}\cdots D_n.
$$

空乘积约定为单位矩阵。这个公式说明，较早位置的 correction 还要承受后续位置的列衰减；
因此不能简单地把 correction 相加，也不能从最后一个候选状态直接截断得到任意接受前缀。

### 4. commit 阶段：反向累积并写回原 checkpoint

采样器返回每个 request 的 `num_accepted_tokens`。该计数包含 leading target token，取值被
限制在 $[0,L]$。对接受长度 $n$，commit kernel 从 $t=n$ 反向遍历到 $1$，维护：

$$
P\leftarrow I_K,\qquad U\leftarrow 0_{V\times K}.
$$

每处理一个位置 $t$，执行

$$
U\leftarrow U+u_t\hat k_t^{\mathsf T}P,
\qquad
P\leftarrow D_tP.
$$

遍历完成后，得到

$$
S_{\rm rebuilt}=S_0P+U.
$$

由于所有 $D_t$ 都是对角矩阵，它们彼此可交换；上式正好等价于第 3 节的闭式表达。实现
中 `decay_acc` 对应 $P$，`update_acc` 对应 $U$，并且只读取前 $n$ 个记录，因此被
拒绝的 suffix 不会参与重建。最后，$S_{\rm rebuilt}$ 被写回该 request 对应的原生
checkpoint slot，供下一轮 verify 的 state copy/index 逻辑继续使用。

### 5. 与 speculative decoding 生命周期的对应关系

1. **Target forward / verify**：每个 target KDA layer 从上一个已提交状态读取 $S_0$，展开
   整条候选链，输出每个位置的结果，并保存 $R_t$。原 checkpoint 保持不变。
2. **Rejection sampling**：MRV1 根据采样结果确定每个 request 的接受长度 $n$。packed KDA
   行通过 speculative request mask 映射到 sampler 行；填充的 sampled ID `-1` 被排除。
3. **State commit**：对每个 request 独立执行反向累积，只重建接受前缀，并把结果写回原生
   state slot。随后释放本轮 pending records。
4. **下一轮**：下一次 state copy 从已写回的 slot 开始，故 KDA 的递推链在数学上仍然是
   连续的；卷积历史则沿原有 accepted-length 逻辑处理。

### 6. 状态存储量的含义

若有 $H$ 个 local heads、候选长度为 $L$，原始方案需要为每个位置保留完整矩阵，量级为
$O(LHVK)$。RecoverSSM 只保留一个 checkpoint 加上向量记录，量级为
$O(HVK+LH(V+2K))$：

$$
\underbrace{HVK}_{\text{checkpoint}}
+\underbrace{LH(V+2K)}_{\text{correction/key/raw-gate records}}.
$$

这解释了 PR 中的存储下降来源。该 PR 当前仍保留原生 serving cache allocation，新增的是
scratch/gather/writeback 路径；上式是逻辑 tensor size，不等同于最终 allocator 或 serving KV
pool 的实际缩减。

### 7. 数值一致性要点

- Q/K 的零向量归一化结果定义为零，非零向量使用实际范数；加入 CUDA 路径中的
  $10^{-6}$ squared-norm bias 会改变小范数训练向量的结果。
- verify 和 commit 必须使用相同的 gate 参数、lower-bound 分支和 raw-gate 记录，否则
  $D_t$ 不一致，反向重建无法得到同一状态。
- verify checkpoint 不变、commit 只写接受前缀，是多轮 speculative decoding 正确性的关键；
  下一轮的初始状态必须来自本轮 commit 后的 slot。





