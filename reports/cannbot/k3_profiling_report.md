# Kimi K3 Prefill/Decode Profiling 负载分析报告

> 数据来源：`C:\Users\Jarvis\Desktop\kimi_insight\profiling`；按 `model-infer-profiling` 与 `model-infer-perf-breakdown` 的 kernel / step / operator 口径分析。

## 口径与限制

- P = `kimik3_A3_prefill`，D = `kimik3_A5_decode`。
- 访存/计算/通信占比使用 `kernel_details.csv` 的 PipeUtilization 单元时间：计算 = MAC/Vector/Scalar，访存 = MTE1/MTE2/MTE3/FixPipe，通信 = COMMUNICATION/HCCL 核。由于流水单元可重叠，这一组比例表示**设备流水负载构成**，不是互斥墙钟时间。
- P/D 的墙钟阶段占比使用 `step_trace_time.csv` 的 `Stage`、`Computing`、`Communication`、`Overlapped`。
- Attention 包含 MLA/FlashMLA/Attention、KDA、CausalConv、RecurrentKda 等；MoE 包含 MoE 路由、Dispatch/Combine、Token Permute/Unpermute、GroupedMatmul/Expert 相关核。分类依据算子名与 Kimi K3 结构代码，未匹配项归为 Other。

## 阶段总览

| 阶段 | kernel 行数 | Stage (ms) | Computing / Stage | Communication / Stage | Overlapped / Stage | Free / Stage |
|---|---:|---:|---:|---:|---:|---:|
| P (prefill) | 63,874 | 10929.388 | 59.2% | 68.7% | 32.1% | 4.2% |
| D (decode) | 146,289 | 2135.552 | 89.4% | 15.2% | 8.5% | 3.9% |

## 访存 / 计算 / 通信负载构成

| 阶段 | 访存流水时间 (us) | 计算流水时间 (us) | 通信核时间 (us) | 访存 | 计算 | 通信 |
|---|---:|---:|---:|---:|---:|---:|
| P (prefill) | 6,754,675 | 6,094,039 | 12,320,975 | 26.8% | 24.2% | 49.0% |
| D (decode) | 2,965,027 | 2,583,644 | 398,043 | 49.9% | 43.4% | 6.7% |

## Attention / MoE 负载与算子耗时

| 阶段 | Attention kernel 时间 (ms) | MoE kernel 时间 (ms) | A/M kernel 中 Attention | A/M kernel 中 MoE | Attention operator 时间 (ms) | MoE operator 时间 (ms) | 占全部 operator Attention | 占全部 operator MoE | A/M operator 中 Attention | A/M operator 中 MoE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P (prefill) | 2024.753 | 1617.692 | 55.6% | 44.4% | 2005.161 | 1617.692 | 25.1% | 20.3% | 55.3% | 44.7% |
| D (decode) | 492.027 | 817.355 | 37.6% | 62.4% | 492.027 | 817.355 | 20.3% | 33.8% | 37.6% | 62.4% |

## 主要算子

### P (prefill)

| 算子 | 类型 | 核心 | 次数 | 总耗时 (ms) | 占 op_statistic |
|---|---|---|---:|---:|---:|
| `GroupedMatmul` | moe | MIX_AIC | 1,104 | 1396.575 | 17.49% |
| `ChunkKdaFwd` | attention | MIX_AIC | 432 | 1333.460 | 16.70% |
| `DequantSituQuant` | other | AI_VECTOR_CORE | 1,104 | 784.558 | 9.82% |
| `QuantBatchMatmulV3` | other | MIX_AIC | 2,424 | 752.520 | 9.42% |
| `DynamicQuant` | other | AI_VECTOR_CORE | 2,976 | 665.931 | 8.34% |
| `FusedInferAttentionScore` | attention | MIX_AIC | 288 | 487.769 | 6.11% |
| `allgatherAicpuKernel` | communication | AI_CPU | 1,248 | 383.625 | 4.80% |
| `MatMulV3` | other | AI_CORE | 984 | 361.993 | 4.53% |
| `reduce_scatterAicpuKernel` | communication | AI_CPU | 1,152 | 332.453 | 4.16% |
| `MatMulV2` | other | AI_CORE | 1,008 | 286.608 | 3.59% |
| `QuantBatchMatmulV3` | other | AI_CORE | 552 | 184.962 | 2.32% |
| `MoeTokenUnpermute` | moe | AI_VECTOR_CORE | 1,104 | 80.338 | 1.01% |
| `_apply_attn_res_kernel` | attention | AI_VECTOR_CORE | 1,128 | 76.120 | 0.95% |
| `MoeInitRoutingV3` | moe | MIX_AIV | 552 | 74.777 | 0.94% |
| `Slice` | other | AI_VECTOR_CORE | 5,790 | 65.669 | 0.82% |

### D (decode)

| 算子 | 类型 | 核心 | 次数 | 总耗时 (ms) | 占 op_statistic |
|---|---|---|---:|---:|---:|
| `GroupedMatmulSituQuant` | moe | MIX_AIC | 3,588 | 405.279 | 16.75% |
| `FlashMlaWithKvcache` | attention | MIX_AIC | 2,067 | 249.041 | 10.29% |
| `GroupedMatmul` | moe | MIX_AIC | 3,588 | 215.590 | 8.91% |
| `SituMxQuant` | other | AI_VECTOR_CORE | 3,588 | 212.784 | 8.79% |
| `QuantBatchMatmulV4` | other | MIX_AIC | 7,176 | 166.162 | 6.87% |
| `QuantBatchMatmulV3` | other | AI_CORE | 9,945 | 157.749 | 6.52% |
| `MatMulV3` | other | AI_CORE | 7,995 | 140.159 | 5.79% |
| `MatMulV3` | other | MIX_AIC | 4,251 | 116.959 | 4.83% |
| `MoeDistributeDispatchV3` | moe | MIX_AIV | 3,588 | 93.162 | 3.85% |
| `MoeDistributeCombineV3` | moe | AI_VECTOR_CORE | 3,588 | 73.356 | 3.03% |
| `RecurrentKda` | attention | AI_VECTOR_CORE | 2,691 | 70.840 | 2.93% |
| `DynamicMxQuant` | other | MIX_AIV | 14,469 | 66.500 | 2.75% |
| `HcclLaunchAicpuKernel` | communication | AI_CPU | 507 | 65.478 | 2.71% |
| `AttnResFwdFused` | attention | AI_VECTOR_CORE | 7,293 | 64.865 | 2.68% |
| `MlaPrologV3` | attention | MIX_AIC | 936 | 38.273 | 1.58% |

## 瓶颈判断与优化建议

### P (prefill)

- 阶段级瓶颈偏通信：通信墙钟占 Stage **68.7%**，高于计算的 **59.2%**；应优先检查 HCCL all-gather/reduce-scatter/all-to-all 的消息量、并行拓扑和通信计算重叠。
- kernel 负载中 Attention 更重（55.6% vs MoE 44.4%）。重点关注 MLA/FlashMLA、KDA/RecurrentKda、Attention residual/prolog，以及 KV cache 访存。
- `op_statistic` 的 Attention 算子耗时占 Attention+MoE **55.3%**，建议先做 FlashMLA/Attention 主核、KV cache 访问和 KDA 相关算子优化。

### D (decode)

- 阶段级瓶颈偏计算：计算墙钟占 Stage **89.4%**；应优先优化高耗时矩阵乘、量化/反量化及 Attention/MoE 主核。
- kernel 负载中 MoE 更重（62.4% vs Attention 37.6%）。重点关注 MoE 路由、Dispatch/Combine、GroupedMatmul、Token Permute/Unpermute，以及其前后的量化核。
- `op_statistic` 的 MoE 算子耗时占 Attention+MoE **62.4%**，与 kernel 结论一致，建议先做 MoE 通信和 GroupedMatmul 融合/量化路径优化。

