# vLLM-Ascend 中 Kimi K3 投机推理与 KDA 状态回滚分析

## 结论先行

当前 vLLM-Ascend 已经把 Kimi K3 的 MTP/投机验证路径接到了 `fla_npu` 的 `recurrent_kda` 算子。PR #17664 已合并，主要工作是用 `fla_npu.ops.ascendc` 替换仓库内置的 KDA AscendC 算子，同时保留 Kimi K3/GLM5Next 原有的 gate、beta、V-first state 和 speculative metadata 语义。

这条路径已经具备实际的 KDA 投机回滚能力：验证阶段得到每个请求的 `num_accepted_tokens`，再利用 `spec_state_indices` 从 state pool 中定位已接受前缀对应的状态，重新执行当前短投机窗口，并把每个 token 的新状态写回指定 slot。

需要区分两种“回滚”含义：

1. **当前 vLLM-Ascend 的实现**：显式维护每个 speculative token 的 recurrent state slot，验证后从 accepted token 对应的 state 恢复，再重算窗口。
2. **Kimi K3 技术报告中的理想优化**：缓存较小的 projected inputs，在片上 replay accepted tokens，避免保存每个 draft position 的完整大状态。

当前代码已经完成第一种机制，但从调用方和 `recurrent_kda` 接口看，不能证明它已经实现了报告中第二种“只缓存 projected inputs”的优化。

## 当前代码状态

### PR #17664

PR [#17664](https://github.com/vllm-project/vllm-ascend/pull/17664) 已于 2026-10-04 合并。PR 的目标是：

- 直接从 `fla_npu.ops.ascendc` 导入 `chunk_kda_fwd` 和 `recurrent_kda`；
- 删除 vLLM-Ascend 仓库内置的 chunk/recurrent KDA AscendC 实现、Torch schema 和构建入口；
- 保留原有 gate/beta 处理、V-first state 和 speculative decoding metadata；
- 调整 recurrent 算子的参数和返回值。

PR 的测试包括 KDA recurrent 集成测试，并报告了 A5 上 recurrent KDA 算子平均延迟从 121.810 微秒降至 94.639 微秒，下降 22.31%；A3 上则从 81.573 微秒变为 83.999 微秒，增加 2.97%。这些是算子级 profile，不等于完整请求的投机推理加速。

主要代码位置：

- [公共 KDA wrapper](C:\Users\ASUS\Desktop\kimi_k3\vllm-ascend\vllm_ascend\ops\kda.py)
- [GLM5Next KDA 适配层](C:\Users\ASUS\Desktop\kimi_k3\vllm-ascend\vllm_ascend\models\glm5next\ops\kda.py)
- [GLM5Next/Kimi 风格 KDA 调用方](C:\Users\ASUS\Desktop\kimi_k3\vllm-ascend\vllm_ascend\models\glm5next\kda.py)
- [fla_npu recurrent KDA README](C:\Users\ASUS\Desktop\kimi_k3\kernels\flash-linear-attention-npu\fla\ops\ascendc\kda\recurrent_kda\README.md)

## 投机推理的完整调用路径

Kimi K3 的 KDA 层包含短卷积和 recurrent KDA 两个有状态阶段。因此投机回滚不是只恢复一个矩阵，而是要同时处理：

```text
MTP/草稿 token
      ↓
投影 q|k|v、gate、beta
      ↓
spec causal-conv
      │ 依据 num_accepted_tokens 恢复/滑动卷积状态
      ↓
得到 q_spec、k_spec、v_spec
      ↓
recurrent_kda
      │ 依据 num_accepted_tokens 选择 KDA 起始 state
      │ 依据 spec_state_indices 为每个 token 选择 state slot
      │ 顺序执行 KDA recurrence 并写回状态
      ↓
把 spec 输出写回原 token 位置
```

在 [KDA caller](C:\Users\ASUS\Desktop\kimi_k3\vllm-ascend\vllm_ascend\models\glm5next\kda.py) 中，投机路径首先读取：

- `spec_query_start_loc`：每条投机序列在 packed token buffer 中的边界；
- `spec_state_indices_tensor`：每条序列的 state slot 表；
- `spec_token_indx`：投机 token 在原始输出 buffer 中的位置；
- `num_accepted_tokens`：本轮验证后每条序列接受的 token 数。

投机 token 与普通 decode token 被分开处理。投机分支使用 `causal_conv1d(..., num_accepted_tokens=...)`，然后调用：

```python
spec_output = recurrent_kda(
    rearrange(q_spec),
    rearrange(k_spec),
    rearrange(v_spec),
    g1_spec,
    beta_spec,
    recurrent_state,
    spec_query_start_loc[: num_spec_decodes + 1],
    spec_state_indices_tensor,
    self.A_log,
    self.dt_bias,
    lower_bound,
    num_accepted_tokens,
)
```

最终通过 `index_copy_` 将投机输出放回原 token 位置。这样可以把投机 token 和非投机 token 混合在同一轮调度中，而不要求输出 buffer 按投机序列重新排列。

## 状态回滚解决的问题

### KDA 的根本问题

KDA 通过递推状态表示历史信息：

\[
S_t = F(S_{t-1}, q_t, k_t, v_t, g_t, \beta_t)
\]

普通 decode 时，状态可以原位更新。但在投机解码中，草稿模型可能一次给出多个 token，目标模型验证后只接受前缀：

```text
草稿 token:       d1 d2 d3 d4 d5
验证结果:         ✓  ✓  ✓  ✗  -
最终接受前缀:      d1 d2 d3
```

如果 KDA 已经顺序执行了 `d1...d5`，状态会停在 `S5`；目标序列真正需要的是 `S3`。传统的原位更新无法通过简单地删除末尾 token 来恢复 `S3`。

### 回滚机制解决的三个问题

1. **状态越过 accepted prefix**：拒绝 token 不应污染后续 KDA 状态。
2. **短卷积状态也会越过 accepted prefix**：KDA 前的 q/k/v causal convolution 同样依赖历史，必须和 recurrent state 使用同一组 accepted metadata。
3. **批量调度中的状态归属**：不同请求、不同 speculative step 不能覆盖彼此的状态，因此需要显式 slot 表和 state pool。

## 当前实现的状态恢复机制

### `num_accepted_tokens`

`num_accepted_tokens` 是每条投机序列的 accepted prefix 长度。GLM5Next 适配层会将它传给 causal convolution 和 `recurrent_kda`。

在 `recurrent_kda` kernel 中，`ResolveInitialStateSlot()` 使用 accepted 数量计算起始 token 位置，随后从 `ssm_state_indices` 取得对应 state slot。其语义是：从已接受前缀末端的状态开始处理当前投机窗口。

### `ssm_state_indices`

`ssm_state_indices` 是 state pool 的索引表：

```text
序列 i:  [slot_before, slot_1, slot_2, ..., slot_m]
                         ↑
                 每个 speculative token 的输出状态
```

kernel 对每个 token 执行 KDA recurrence 后，把结果写回该 token 对应的 slot。这样拒绝的草稿状态可以留在独立 slot 中，不会直接覆盖已接受序列的主状态；下一轮调度只需选择正确的 accepted slot。

### 状态写回

`recurrent_kda` 以 `inplace_final_state=True` 和 `state_v_first=True` 调用。它不是返回一个独立的最终状态再由 Python 复制回 cache，而是直接更新传入的 state pool。输出和状态写回分别通过 recurrent kernel 的 attention output 与 `CopyOutState` 路径完成。

## 当前实现与技术报告方案的差异

技术报告中的优化思路是：

```text
不保存每个 draft position 的完整 S_t
只缓存 projected inputs
验证后从 S_accepted 开始片上 replay
写回 verified/bonus token 状态
```

当前 vLLM-Ascend 代码体现的是：

```text
显式 state pool
每个 speculative token 一个 state slot
accepted 数量定位起始 slot
直接读取 q/k/v/g/beta 并顺序重算
把每个 token 状态写回对应 slot
```

因此，当前实现已经具备**功能上的状态回滚**，但其状态内存策略更接近“多状态槽快照方案”。`recurrent_kda` 的接口没有 `projected_inputs` 专用缓存参数，也没有报告中所说的 replay-only 分支；q/k/v/g/beta 仍由调用方作为当前投机窗口的完整输入传入。

这一区分很重要：

- `num_accepted_tokens + state_indices` 证明了它支持 MTP rejection rollback；
- 但不能据此声称已经获得技术报告中“避免每个 draft state 快照”的全部内存流量收益。

## 对系统性能的收益

### 已经可以确认的收益

当前状态槽方案的主要收益是正确性和调度效率：

- 拒绝 token 不会污染已接受前缀的 KDA state；
- 不需要回滚整个长上下文，只需处理最多几个 speculative token；
- 同一 recurrent kernel 可以处理短窗口，接口限制每段 recurrent 序列长度不超过 8；
- speculative 与普通 decode 可以在同一批次中分流执行；
- state pool 允许不同请求使用不同 slot，减少 Python 侧的显式 gather/scatter。

PR #17664 的 A5 profile 显示，替换为 fla_npu 后 recurrent KDA 算子平均延迟下降约 22.3%。这说明算子实现本身有收益，但不能把这 22.3% 直接等同于投机回滚收益，也不能推导出完整请求吞吐提升。

### 理论上的状态流量收益

设：

- (H_v)：value head 数；
- (K,V)：KDA state 的两个矩阵维度；
- (d_s)：state dtype 字节数；
- (d_q,d_g,d_b)：q/k/v、gate、beta 的每元素字节数。

一个完整 KDA state 的大小近似为：

\[
B_S = H_v \times V \times K \times d_s
\]

一个 token 的 projected inputs 大小近似为：

\[
B_P = H_qK d_q + H_qK d_q + H_vV d_q + H_vK d_g + H_v d_b
\]

以常见的 (K=V=128)、BF16 q/k/v、FP32 gate/beta、FP32 state、且 (H_q=H_v=H) 为例：

\[
B_S \approx 4H \times 128^2 = 65536H\text{ bytes}
\]

\[
B_P \approx 2H(128+128+128)+4H(128+1)=1280H\text{ bytes}
\]

因此：

\[
\frac{B_S}{B_P} \approx 51.2
\]

如果把传统 snapshot 方案的状态读写粗略按一次写入加一次恢复读取计算，则仅从状态搬运角度看，projected-input replay 的潜在流量可以降低约：

\[
1 - \frac{B_P}{2B_S} \approx 99.0\%
\]

这只是局部内存流量上界，不能直接转化为端到端 99% 加速。实际收益还会受到以下部分限制：

- `S @ k`、`outer(delta,k)`、`S @ q` 的计算量；
- gate/beta、短卷积和输出写回；
- NPU kernel 启动和同步；
- state pool 的 slot 写回；
- batch 中非投机请求的比例。

如果 state 使用 BF16，上述 state/input 比例约减半，仍然约为 25.6 倍，说明 state snapshot 的内存压力通常远大于 projected-input 缓存。

### 投机吞吐的理论估计

设每轮草稿窗口长度为 (m)，平均接受 token 数为 (a)，bonus token 使每轮平均产出 (a+1) 个 token；设：

- (C_t)：目标模型普通单 token decode 成本；
- (C_d)：草稿模型生成一个 token 的成本；
- (C_v(m))：目标模型验证整个投机窗口的成本；
- (C_r(a,m))：回滚/状态重算额外成本。

普通 decode 每产出一个 token 的成本约为 (C_t)。投机路径每轮的平均成本约为：

\[
C_{spec}=mC_d+C_v(m)+C_r(a,m)
\]

对应的理想吞吐倍率为：

\[
Speedup \approx \frac{(a+1)C_t}{mC_d+C_v(m)+C_r(a,m)}
\]

这个公式说明：

- 接受率越高，(a) 越大，投机收益越高；
- 草稿模型越便宜，(C_d) 越小，收益越高；
- KDA 回滚越接近短窗口重算，(C_r) 越小，收益越高；
- 如果接受率低或 state snapshot 搬运很重，投机可能退化甚至变慢。

举一个明确标注假设的例子：若 (m=4)、平均接受 (a=3)、草稿模型单 token 成本为 (0.25C_t)、验证成本为 (1.0C_t)、回滚成本为 (0.25C_t)，则：

\[
Speedup \approx \frac{4C_t}{4\times0.25C_t+1.0C_t+0.25C_t}
 = \frac{4}{2.25}\approx1.78\times
\]

这只是模型化估计，不是当前 vLLM-Ascend 的实测结果。真实结果需要同时测量接受率、draft/target 比例、state slot 读写、conv 回滚和混合 batch 调度。

## 当前实现的限制与风险点

1. **短窗口限制**：`recurrent_kda` 当前每条 recurrent 序列最多处理 8 个 token，适合 MTP decode，不适合直接承担长序列 recurrent prefill。
2. **state slot 依赖 metadata 正确性**：`spec_state_indices_tensor`、`spec_query_start_loc` 和 `num_accepted_tokens` 必须来自同一轮调度；任何错位都会把一个请求的 state 读给另一个请求。
3. **状态内存仍可能较大**：当前方案保留多个 speculative state slot，尚未体现报告中 projected-input replay 的完整节省。
4. **causal-conv 与 recurrent KDA 必须一致回滚**：只恢复 KDA state 而不恢复 q/k/v 短卷积状态，会产生不一致的输入投影。
5. **算子 profile 不等于服务吞吐**：PR 中 A5 的 recurrent 算子下降 22.31%，但完整请求延迟还受 MoE、MLA、调度、通信和 draft acceptance 影响。
6. **A3/A5 结果不同**：PR 报告 A3 recurrent 算子平均延迟略有上升，因此不能假设所有 Ascend 平台都获得相同收益。

## 建议的验证矩阵

要确认完整的 Kimi K3 投机回滚收益，至少应记录以下指标：

| 指标 | 目的 |
|---|---|
| 平均 draft window 长度 (m) | 确认实际投机窗口规模 |
| 平均 accepted token 数 (a) | 估计投机有效率 |
| 接受率 (a/m) | 判断回滚是否频繁 |
| `recurrent_kda` kernel latency | 测量算子本身 |
| causal-conv latency | 测量另一段有状态回滚 |
| state pool 读写字节数 | 判断 snapshot 成本 |
| 每 token 端到端延迟 | 计算真实吞吐收益 |
| TTFT/TPOT | 区分首 token 和持续生成影响 |
| 混合 batch 中 speculative/non-spec 比例 | 判断调度干扰 |
| 回滚前后 state 一致性 | 验证正确性 |

建议做三组基线：

1. 普通单 token decode；
2. 当前 state-slot rollback；
3. 理想 projected-input replay 或等价的低状态流量实现。

只有第 2 组与第 3 组的对比，才能判断技术报告中“减少状态 traffic”的优化是否已经在当前 vLLM-Ascend 集成中实现。

## 最终判断

当前 vLLM-Ascend 的 Kimi K3 投机推理已经不是“只有接口预留”：`num_accepted_tokens`、causal-conv rollback、`spec_state_indices`、state pool 和 `recurrent_kda` 调用形成了完整的 MTP 状态恢复链路。

它解决的是 KDA 原位递推状态无法在拒绝草稿后直接截断的问题，收益是保证状态正确性、把回滚范围限制在短 speculative window，并利用 fused recurrent kernel 降低执行开销。

但从实现细节看，当前路径更接近**显式 state-slot 快照恢复方案**。技术报告中的**projected-input replay 方案**可以作为后续优化方向：它有望把状态相关内存流量降低一个数量级以上，但其端到端收益仍需结合接受率、验证成本和 NPU kernel 实测确认。

## 未来演进方向

### 1. 接入 vLLM 的动态投机部署

vLLM 主仓当前已经提供 batch-size 驱动的 Dynamic Speculative Decoding。部署时仍通过 `--speculative-config` 配置投机方法和最大 draft 长度，同时增加 `num_speculative_tokens_per_batch_size`：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "eagle3",
    "model": "<draft-model>",
    "num_speculative_tokens": 5,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

每一项 `[start_bs, end_bs, K]` 表示并发 batch size 落在该区间时使用 `K` 个 draft token；`K=0` 表示在高并发时关闭投机。vLLM 的设计动机是：验证成本近似随 `batch_size × K` 增长，当并发升高到某个临界点后，继续投机会恶化 TPOT，因此应动态减小 K。

当前部署特征：

- vLLM Model Runner V2 可以保留 full CUDA graph；
- Model Runner V1 在运行时改变验证长度，通常会降级到 piecewise graph；
- data parallel 当前会自动关闭这项动态 schedule，避免不同 DP rank 选择不同 K 导致 collective divergence；
- V1 只对部分方法开放动态 K，主线支持范围包括 Eagle/Eagle3、MTP、DFlash 等，具体还受版本和 runner 限制；
- vLLM 还在推进按请求分配不同 draft 长度的 packed variable-length speculative decoding，目前成熟度低于 batch-level schedule。

参考：

- [vLLM Dynamic Speculative Decoding 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md)
- [vLLM 动态投机配置与 graph 处理](C:\Users\ASUS\Desktop\kimi_k3\vllm\vllm\config\vllm.py:1041)
- [SpeculativeConfig 动态 schedule 字段](C:\Users\ASUS\Desktop\kimi_k3\vllm\vllm\config\speculative.py:494)

#### 对 Kimi K3 的适配

Kimi K3 当前的 recurrent kernel 已支持 packed `cu_seqlens`、speculative state indices 和最多 8 token 的短窗口，因此动态 K 可以沿着以下路径接入：

1. 第一阶段采用**批次级动态 K**：一个 decode batch 根据并发量选择统一 K，K 限制在 `0..8`，并同步调整 `spec_query_start_loc`、conv state 宽度和 `spec_state_indices`。
2. 第二阶段支持**请求级动态 K**：同一批请求使用不同的 K，由 scheduler 生成变长 `cu_seqlens`，将每条请求的状态列数限制为实际验证长度。
3. 第三阶段把 `num_accepted_tokens` 纳入 K 选择策略：低接受率请求减少 K，高接受率请求增加 K；高并发时优先降低 K，低并发或长尾 RL rollout 时提高 K。

K 的选择不能只看 batch size，还应考虑 KDA 回滚成本：

\[
K^* = \arg\max_K \frac{E[A_K]+1}{C_{draft}(K)+C_{verify}(K)+C_{rollback}(K)}
\]

其中 (E[A_K]) 是窗口长度 K 下的平均接受 token 数。当前 state-slot 方案中，增大 K 会同时增加 state slot 数、状态写回量和短卷积缓存宽度；因此 K 的最优值可能低于普通 Transformer 草稿模型的最优值。

#### 对 Kimi K3 的主要工程改动

- 让 vLLM-Ascend 的 scheduler 将动态 K 转换成 KDA 可消费的 `cu_seqlens` 和二维 `spec_state_indices`；
- 为每个 K tier 准备 graph/capture shape，避免动态 K 破坏 NPU graph replay；
- 将 `state_capacity`、slot 生命周期和 rejected slot 回收纳入调度器；
- 确保 causal-conv 与 recurrent KDA 使用同一个 accepted prefix 和同一组 token offsets；
- 将 acceptance、K、state traffic、TPOT 作为联合反馈，而不是只按并发量静态查表。

### 2. 引入投机推理 KV 滑窗

vLLM-Ascend 的 PR #10023 和 PR #14596 已经形成了一套较完整的 draft KV sliding window 方案：**只限制 draft 模型读取的 KV 范围，target 模型仍然读取完整上下文**。因此滑窗不会改变最终 target 的计算语义，主要用于恢复长上下文下 draft 的接受率。

#### PR #10023：Eagle3 / DFlash

PR [#10023](https://github.com/vllm-project/vllm-ascend/pull/10023) 为 Eagle3 和 DFlash 增加了 `draft_window_size`。实现要点是：

- 根据最终序列长度和窗口 W 计算窗口起点；
- 将起点按 KV block 对齐；
- 裁剪 draft attention 使用的 block table 和 sequence length；
- 保留完整 block table，用于绝对 slot mapping 和 KV cache 写入；
- 使用固定形状的窗口 block table，兼容 graph capture/replay。

PR 的长上下文结果显示，滑窗可以显著改善接受率：

- Eagle3 LongBenchV2 平均接受长度从 1.3358 提高到 1.8809；
- DFlash 在 32k 输入上平均接受长度从 3.5548 提高到约 4.3646；
- DFlash 输出 token/s 在 32k、64k、96k、128k 输入上分别约提升 11.0%、3.6%、17.6%、15.5%（窗口取 2048）；
- PR 总结的 Eagle3 输出性能收益约为 30%~48%，DFlash 约为 10%，具体数值依赖任务和窗口大小。

这说明滑窗的主要收益不是减少 target 计算，而是避免 draft 在超出训练上下文后出现接受率崩溃。

#### PR #14596：DSpark

PR [#14596](https://github.com/vllm-project/vllm-ascend/pull/14596) 将同一能力扩展到 DSpark，并处理了 DSpark 的 per-group block table：

- 在 DSpark 的每个 attention group 上分别应用窗口；
- 只裁剪 FIA 实际读取的 KV-length tensors；
- 保持 slot mapping 使用完整、绝对地址的 block table；
- 对 MTP 强制关闭滑窗，因为 MTP 复用 target 模型层，改变其 attention 窗口会改变 target 计算；
- 对原生训练时已经使用 sliding window 的 DeepSeek-V4/DSpark draft 给出兼容性提示；
- MRV1 需要在 common metadata 和 DSpark per-group metadata 的正确位置应用窗口，避免重复 windowing；
- MRV2 需要在 builder 输入层做窗口化，并注意 graph replay 和 `enforce_eager` 约束。

#### 滑窗为什么适合长上下文 draft

很多 draft 模型只在 4k~8k 上下文训练。输入变成几十万 token 后，如果 draft 仍然对完整前缀做 attention，它看到的分布已经偏离训练分布，后续 token 接受率快速下降。滑窗把 draft 的有效上下文限制在最近 W 个 token，使 draft 的输入分布回到训练范围；target 仍保留完整上下文，因此不会改变最终输出的正确性保证。

窗口大小不应盲目取小：W 小于 draft 原生训练窗口会丢掉本来有用的上下文。vLLM-Ascend 文档建议在 512~4096 范围内用真实任务测试，并观察 `SpecDecoding metrics: Mean acceptance length`。

### 3. KV 滑窗与 Kimi K3 的结合方式

#### 当前 MTP 路径不能直接套用 draft KV 滑窗

Kimi K3 的当前投机路径是 MTP/target-layer reuse。MTP 草稿阶段复用 target 模型自身的层，因此像 PR #14596 所描述的那样，直接缩短 MTP 的 attention 窗口会改变 target 的计算，不能简单视为“只优化 draft”。这也是现有 vLLM 文档对 MTP 强制关闭 draft window 的原因。

此外，KDA 的主状态不是普通 token-by-token KV cache，而是 recurrent state。若直接把 target KDA 状态截断为最近 W 个 token，需要从窗口起点重新构建状态：

\[
S_{t_0} = F^{t_0}(S_{0}, x_{1:t_0})
\]

这会重新引入长前缀计算，抵消滑窗带来的收益。因此，滑窗应优先作用于**独立 draft 模型的 attention KV**，而不是直接截断 target KDA state。

#### 推荐的 Kimi K3 演进方案

1. **独立轻量 draft + KDA target**：为 Kimi K3 配置 Eagle3、DFlash 或 DSpark 类独立 draft，target 保持完整 KDA/MLA 上下文；复用 PR #10023/#14596 的 `draft_window_size`。
2. **KDA-aware draft window**：如果 draft 自身也包含 KDA，窗口边界不能只裁 block table，而要同时维护 draft 的 recurrent state checkpoint；窗口移动时从最近 checkpoint replay 到窗口起点。
3. **滑窗与动态 K 联动**：长上下文时先通过窗口提高 acceptance，再根据 batch size 和实时 acceptance 选择 K；窗口 W 与 K 不应独立调参。
4. **状态绝对索引保持不变**：滑窗只改变 draft attention 的读取范围，`spec_state_indices`、target KDA state 和 rejected-slot 回收仍使用全局 token/slot 坐标。

联合目标可以写为：

\[
(W^*,K^*) = \arg\max_{W,K}
\frac{E[A(W,K)]+1}
{C_{draft}(W,K)+C_{verify}(K)+C_{rollback}(K)}
\]

其中：

- 增大 W 可能提高接受率，但增加 draft attention 计算和 KV 读取；
- 减小 W 可能降低 draft 计算，但如果低于训练窗口会降低接受率；
- 增大 K 可能提高每轮产出 token 数，但会增加验证、回滚和 state slot 流量；
- KDA 的 `C_rollback` 还受到 state snapshot 或 projected-input replay 策略影响。

### 4. 推荐的落地顺序

对 Kimi K3，建议按下面顺序推进：

1. **先补齐当前实现的指标**：记录 acceptance length、K、窗口长度、recurrent/conv kernel latency、state slot 读写和 TPOT。
2. **先做 batch-level dynamic K**：利用 vLLM 已有 schedule，限制 K≤8，并在 NPU graph 中为每个 K tier 预留 shape。
3. **再做独立 draft 的 KV sliding window**：优先支持 Eagle3/DFlash/DSpark draft，target KDA 保持全上下文。
4. **再做 request-level variable K**：让 `cu_seqlens` 和 `spec_state_indices` 支持同一 batch 内的异构窗口长度。
5. **最后优化 KDA rollback traffic**：把当前 state-slot 方案演进到 projected-input replay，减少每个 draft position 的完整 state 读写。

这样可以把三个收益来源分开验证：

- dynamic K：减少高并发时的无效验证开销；
- draft KV window：恢复长上下文下的接受率；
- projected-input replay：降低 KDA 回滚的状态内存流量。
