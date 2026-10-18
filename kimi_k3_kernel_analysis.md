# Kimi K3 Attention Kernel 分析

> 分析范围：`kernels/ops-transformer/attention/flash_mla`、`flash_mla_metadata`、`chunk_kda_fwd`、`mla_prolog_v3`、`block_attn_res_prepare`、`block_attn_res_update`、`recurrent_kda`。用户消息中 `chunk_kda_fwd` 重复出现一次，以下合并分析。

## 1. 总体关系

这 7 个算子可按模型执行链分为三组：

| 组 | 算子 | 作用 |
|---|---|---|
| MLA | `mla_prolog_v3 -> flash_mla_metadata -> flash_mla` | 从输入 hidden state 生成 MLA 的 Query/压缩 KV/RoPE cache，再对分页 KV 做注意力 |
| Gated Delta/线性注意力 | `chunk_kda_fwd`、`recurrent_kda` | 前者按 chunk 并行处理训练/长序列，后者按 token 递归处理 decode/MTP 短序列 |
| Attention Residuals | `block_attn_res_prepare -> block_attn_res_update` | 先聚合历史残差块的 online-softmax 统计量，再并入当前残差并生成下一层输入 |

共同的软件分层是：OpDef/InferShape（注册与形状） -> aclnn 两段式接口（校验、连续化、workspace/executor） -> L0/launcher -> AscendC device kernel；`flash_mla_metadata` 例外地在 AICPU 上生成调度元数据。

## 2. flash_mla：非量化分页 MLA 注意力

### 功能与数学定义

`flash_mla` 面向 Ascend 950，输入的 q 和 k_cache 已经把 MLA 的 nope(512) 与 rope(64) 拼成 576 维；算子不执行 KV 压缩或 RoPE。k_cache 同时承担 K、V：K 使用 576 维，V 取前 `head_dim_v=512` 维。

对每个 batch，按 `block_table` 将逻辑页映射到物理 KV 页：

```text
K = k_cache[..., :576]
V = k_cache[..., :512]
S = softmax_scale * QK^T + mask
O = softmax(S) V
LSE = log(sum(exp(S)))       # return_softmax_lse=true 时输出
```

### 接口和关键约束

- q：FP16/BF16，当前主路径为 TND，形状 `(Q_T,Q_N,576)`；Q 头数支持 64/96。
- k_cache：分页 `PA_BBND (num_blocks,128,1,576)` 或 `PA_NZ (num_blocks,1,36,128,16)`；block size 固定 128，NZ 内块 D0=16。
- block_table：`(B,max_num_blocks_per_seq)` 的 INT32 物理页表；cache_seqlens：`(B,)`。
- `cu_seqlens_q`：TND 必传，首项 0、末项 Q_T；`seqused_q` 可给每 batch 实际 query 长度。
- mask_mode 仅 0（无 mask）和 3（右对齐因果 mask）；mode=3 时 attn_mask 为 `(2048,2048)` INT8。
- 输出 attn_out 为 NTD、`(Q_N,Q_T,512)`；可选 FP32 softmax_lse 为 `(Q_N,Q_T)`。
- metadata 必须由配套 `flash_mla_metadata` 按同一 batch、头数、head_dim、mask/layout 重新生成；输入变化而复用旧 metadata 会导致错误或非法访问。

### 实现结构

`op_host/flash_mla_def.cpp` 注册输入、属性和 Ascend950 AICore 配置；`flash_mla_infershape.cpp` 根据 `head_dim_v+64=576` 和 layout 推导输出。aclnn 包装在 `op_api/aclnn_flash_mla.cpp` 中完成参数校验、连续化和 inner op 调用。

device 入口 `op_kernel/flash_mla.cpp` 读取 tiling，构造 FlashMla 对象并调用 `Init/Process`。arch35 实现把工作拆成：

1. Cube：加载 Q/K，执行 QK^T 和 P·V 矩阵乘（`flash_mla_block_cube.h`）。
2. Vector：应用 mask、在线 softmax、跨 S2 block 合并，并可写出 LSE（`flash_mla_block_vec.h`）。
3. 分页地址与 layout：`FlashMlaSeqLensTool` 解析 TND/BSND/BNSD 的实际长度，offset calculator 按 block_table 读取 PA_BBND/PA_NZ。
4. 流水：L1/L0/UB 双缓冲、AIC/AIV 协作；softmax 采用 max/sum 在线折叠，避免保存完整注意力矩阵。

## 3. flash_mla_metadata：Flash MLA 的调度元数据

### 功能

该算子不做注意力数值计算，而是根据 query/KV 实际长度、头数和硬件 AIC/AIV 核数，生成主算子消费的负载均衡 schedule。典型顺序是：

`flash_mla_metadata(...) -> metadata -> flash_mla(..., metadata=metadata)`

### 参数和校验

- 必须提供 `cache_seqlens`，batch size 由其长度推导。
- TND 时必须提供 `cu_seqlens_q (B+1)`；BSND/BNSD 时禁止提供。
- MLA 固定 `num_heads_kv=1`、`head_dim_qk=576`、`head_dim_v=512`；mask_mode 仅 0/3。
- metadata 容量必须覆盖当前 section 数和设备核数。

### AICPU 实现

`op_kernel_aicpu/flash_mla_metadata_aicpu.cpp` 的 `Compute` 顺序为：

`Prepare -> BalanceSchedule -> CheckMetadataCapacity -> GenMetadata`

其中 `Prepare` 读取累计 query 长度、逐 batch cache 长度和可选 seqused_q；`BalanceSchedule` 使用 `section_stream_k` 负载均衡器；`GenMetadata` 写出三段固定布局：

```text
[16-word HEAD]
[sectionNum * aicCoreNum * 16-word FA records]
[sectionNum * aivCoreNum * 16-word FD records]
```

`flash_mla_metadata.h` 定义 FA/FD 字段索引（例如 BN2/M/S2 起止、workspace index/count）和 16-word stride。HEAD 记录 section 数、AIC/AIV 核数、tile 大小及输出布局。该固定 ABI 必须与 flash_mla kernel 的 metadata 解析保持一致。

## 4. chunk_kda_fwd：分块 Gated Delta Rule 前向

### 功能和公式

`chunk_kda_fwd` 面向不涉及 Context Parallel 切分的 KDA 前向，按 `chunkSize=C` 将序列划分为 M 个 chunk，同时输出 attention、final state 和可选反向中间量。支持 BSND/BNSD/TND/NTD、变长序列、GVA（HV>H）和 `state_v_first`。

gate 在 kernel 内可选激活：

```text
gamma = g                                  (use_gate_in_kernel=false)
gamma = -exp(A_log)*softplus(g+dt_bias)    (safe_gate=false)
gamma = lower_bound*sigmoid(exp(A_log)*(g+dt_bias)) (safe_gate=true)
gk = cumsum(gamma within chunk) / ln(2)
```

每个 chunk 先生成：

```text
qg = q * 2^gk
kg = k * 2^(gk_last-gk)
Aqk[i,j] = 1(j<=i) * scale * sum(q_i*k_j*2^(gk_i-gk_j))
L[i,j] = 1(j<i) * beta_i * sum(k_i*k_j*2^(gk_i-gk_j))
Akk = (I+L)^(-1)
w_seed = beta*k*2^gk;  u_seed = beta*v
w = Akk*w_seed; u = Akk*u_seed
v_new = u - w*h_prev
h_next = 2^gk_last*h_prev + kg^T*v_new
attn_out = scale*qg*h_prev + Aqk*v_new
```

### 内部阶段与实现

同一个 L0/device entry 内依次组织 Gate Cumsum、Prepare、Post-WU、FwdH、Finalize；这些不是独立公开算子。Prepare 的 `Akk` 求逆、seed 和 score 使用 FP32；跨 chunk 状态、W/U 和最终状态也保持 FP32，最后写回 q dtype。

- arch22/arch35 各自维护私有 KDA FwdH。
- A5 key2（chunk=64,K=V=128）有 regbase 和 dense 融合模板；满足条件时可把 Gate/Prepare/Post-WU/FwdH 流水融合，否则走通用或 tail/varlen 路径。
- GVA 通过 `floor(h_v/(H_v/H))` 映射 value head 到 q/k head。
- `hCompute`（内部 head-major）与公开 `hOut`（sequence-major）分离；可选输出由非空指针驱动，便于 autograd 重计算。

公开 aclnn 接口采用两段式 `GetWorkspaceSize -> aclnnChunkKdaFwd`，输出列表包括 attnOut、finalState、gk、Aqk、Akk、w、u、qg、kg、vNew、h；Python 层根据 `output_final_state/disable_recompute/return_intermediate_states` 决定哪些实际导出。

## 5. mla_prolog_v3：MLA 前处理融合

### 功能

这是 DeepSeek MLA 的前处理融合算子，将输入 `token_x` 一次性变成 Query、Query-RoPE，并原地更新 KV/ KR cache，同时覆盖多种权重量化、cache 量化和 query 动态量化场景。

核心公式：

```text
cQ   = qcQrScale * RmsNorm(x * W_DQ)
qC   = cQ * W_UQ
qN   = qC * W_UK
qR   = ROPE(cQ * W_QR)
cKV  = kcScale * RmsNorm(x * W_DKV)
kC   = Cache(cKV)
kR   = Cache(ROPE(x * W_KR))
```

可选输出 query_norm=cQ；量化场景还输出 rowmax/groupmax 得到的 dequant scale。W_UQ 与 W_QR、W_DKV 与 W_KR 在输入中横向拼接，减少 matmul 次数。

### Kernel 组织和流水

`op_kernel/mla_prolog_v3.cpp` 根据 tiling key 实例化模板，关键计算阶段为：

1. MatmulCq（x·W_DQ）与 RMSNormCq；
2. MatmulCkvKr（x·[W_DKV|W_KR]）与 RMSNormCkv；
3. MatmulQcQr（cQ·[W_UQ|W_QR]）；
4. RopeQr、RmsNormRopeScatterCkvKr；
5. MatmulQn（cQ·W_UK）并写 query；
6. 动态量化、cache scatter。

AIC/Cube 和 AIV/Vector 之间用 cross-core flags；由于不同阶段切分轴不同，还需要全 Cube/全 Vector barrier。TilingKey 编码 cache mode、BF16/量化场景、quant mode、actualSeqLen、空 tensor、CV 比例、RoPE 开关等。

cache 支持 ND、PA_BSND、PA_NZ、PA_BLK_BSND、PA_BLK_NZ；`cache_index`/actualSeqLen 决定写入位置。A2/A3 与 Ascend950 的量化支持范围不同，必须遵守 README 中的 quant/cache mode 交叉约束。

## 6. block_attn_res_prepare：历史残差注意力阶段一

### 功能和公式

对全部目标 slot 并行计算历史残差块的 attention，输出 online softmax 的三元组 `(numerator, logit_max, exp_sum)=(O,M,L)`：

```text
R[t,n] = sqrt(mean(blockRes[t,n,:]^2) + eps)
Z[s,t,n] = dot(pseudoQuery[s], blockRes[t,n]) / R[t,n]
M[s,t] = max_n Z[s,t,n]
L[s,t] = sum_n exp(Z[s,t,n]-M[s,t])
O[s,t,:] = sum_n exp(Z[s,t,n]-M[s,t]) * blockRes[t,n,:]
```

`valid_blocks[0]` 将 N 截断为有效历史块数；为 0 时直接输出 O=0、L=0、M=float32 最小有限值。

输入为 FLOAT `block_res[T,N,D]`、`pseudo_query[S,D]` 和 UINT64 `valid_blocks[1]`；输出为 FLOAT `numerator[S,T,D]`、`logit_max[S,T]`、`exp_sum[S,T]`。约束为 N<=64、D<=8192、连续 ND tensor。

### 实现

aclnn 两段式接口负责 workspace/executor。device kernel 按 tiling 在 T/S/D 方向切块，Vector 侧做 RMS、点积、max-shift exp 和归约；输出统计量保留 FP32，供阶段二稳定合并。

## 7. block_attn_res_update：当前残差融合阶段二

### 功能和公式

该算子接收阶段一的 O/M/L，将 `delta` 加到可原地更新的 `partialBlockRef`，重新计算当前残差 score，并与历史 online softmax 合并：

```text
p[t,d] = partialBlockRef[t,d] + FP32(delta[t,d])
r[t] = sqrt(mean(p[t,:]^2)+eps)
score[t] = dot(p[t,:], pseudoQuery) / r[t]
m[t] = max(logitMax[t], score[t])
a = exp(logitMax[t]-m[t]); b = exp(score[t]-m[t])
ell[t] = expSum[t]*a + b
updatedNumerator[t,d] = numerator[t,d]*a + p[t,d]*b
h[t,d] = BF16(updatedNumerator[t,d]/ell[t])
partialBlockRef <- p
```

输入 partialBlockRef/delta/pseudoQuery/numerator/logitMax/expSum，输出 BF16 `h[T,D]`；partialBlockRef 是 FLOAT、连续、原地更新，delta 是 BF16。eps 必须为正有限数。

arch35 的 `BlockAttnResUpdateFullD` 将计算拆为两个 Vector phase：phase1 完成 delta 累加、RMS 和 score；phase2 完成 online-softmax 合并、归一化和写回，并用 MTE2/MTE3 双缓冲及尾块 mask 支持 D 非 32/64 对齐场景。

## 8. recurrent_kda：递归 Gated Delta Rule

### 功能和数学定义

`recurrent_kda` 面向 decode/MTP 短序列，每个 token 递归推进状态矩阵，不构造 chunk 内三角矩阵。状态为 `S_t[V_dim,K_dim]`，多个 value head 可共享 query/key head（要求 HV 能被 Hk 整除）。

```text
q_bar = normalize(q) * scale       # 可选 normalize
k_bar = normalize(k)               # 可选 normalize
ell = gate                            # 或 raw gate 经 A_log/dt_bias/safe_gate 转换
d = exp(ell)
beta_bar = beta                     # 或 sigmoid(beta), allow_neg_eigval 时乘2
S_bar = S_prev * d                  # d 沿 K 维广播
delta = beta_bar * (v - S_bar*k_bar)
S_t = S_bar + outer(delta, k_bar)
out_t = S_t*q_bar
```

支持 BSND/TND、`cu_seqlens` 变长序列、`ssm_state_indices` state pool 槽位复用，以及 `num_accepted_tokens` 的 speculative/MTP 选择。initial_state 可为 BF16/FP32，`state_v_first` 控制 `[V,K]` 或 `[K,V]` 存储；`inplace_final_state` 可直接写回 state pool。

### 实现

`op_kernel/recurrent_kda.cpp` 是轻量入口：读取 `RecurrentKdaTilingData`，构造 `RKDA`，将 state output 指向 initialStateOut 或 finalState，再执行 `Init/Process`。arch22/arch35 头文件承载实际 tile、状态更新和同步逻辑；host tiling 根据布局、序列数、head 映射和可选分支选择路径。q/k/v 固定 BF16，gate/beta 支持 FP32/BF16/FP16；A_log、dt_bias 为 FP32。

## 9. 选择与组合建议

- MLA prefill/decode：先用 `mla_prolog_v3` 更新压缩 KV/RoPE cache；TND/分页场景先生成 `flash_mla_metadata`，再调用 `flash_mla`。
- 长序列 KDA：使用 `chunk_kda_fwd`，可导出中间量用于反向或重计算。
- decode/MTP KDA：使用 `recurrent_kda`，通过 state pool 和 ssm_state_indices 避免重复搬运状态。
- Attention Residuals：严格按 `prepare -> update`，阶段一的 FP32 O/M/L 是阶段二在线合并的输入。
- 所有算子都依赖两段式接口；workspace 大小、executor 和 tiling 必须针对当前 shape/layout/属性重新获取。

## 10. 主要源码入口

- Flash MLA：`flash_mla/op_host/flash_mla_def.cpp`、`flash_mla/op_host/flash_mla_infershape.cpp`、`flash_mla/op_kernel/flash_mla.cpp`、`op_kernel/arch35/flash_mla_block_{cube,vec}.h`
- Metadata：`flash_mla_metadata/op_host/op_api/aclnn_flash_mla_metadata.cpp`、`flash_mla_metadata/op_kernel_aicpu/flash_mla_metadata_aicpu.cpp`、`flash_mla_metadata.h`
- Chunk KDA：`chunk_kda_fwd/docs/ChunkKdaFwd算子设计介绍.md`、`op_kernel/chunk_kda_fwd.cpp`、`op_kernel/arch35/*`
- MLA Prolog：`mla_prolog_v3/docs/MlaPrologV3算子设计介绍.md`、`op_kernel/mla_prolog_v3.cpp`、`op_host/mla_prolog_v3_infershape.cpp`
- Residual：`block_attn_res_prepare/op_kernel/block_attn_res_prepare.cpp`、`block_attn_res_update/op_kernel/block_attn_res_update.cpp`、`op_kernel/arch35/block_attn_res_update_full_d.h`
- Recurrent KDA：`recurrent_kda/op_kernel/recurrent_kda.cpp`、`op_kernel/arch22/recurrent_kda_arch22.h`、`op_kernel/arch35/recurrent_kda_arch35.h`

