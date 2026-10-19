# Kimi K3 P/D Profiling 分析

数据目录：`C:\Users\Jarvis\Desktop\kimi_insight\profiling`

- P：`kimik3_A3_prefill/ASCEND_PROFILER_OUTPUT`
- D：`kimik3_A5_decode/.../ASCEND_PROFILER_OUTPUT`

## 统计口径

1. 阶段墙钟时间、计算和通信取 `step_trace_time.csv`。通信同时给出总通信时间和未被计算覆盖的通信时间；因为通信和计算可重叠，所以这些百分比不能直接相加。
2. 访存/计算取 `kernel_details.csv` 的硬件子计数器：访存为 AICore/AIV 的 MTE 时间，计算为 MAC、标量和向量计算时间。该口径反映活跃计算引擎内部的资源忙碌比例，不是阶段墙钟占比。
3. Attention/MoE 取 `op_statistic.csv` 的 `Total Time(us)` 汇总。Attention 归类包括 FlashMLA、MLA prolog、KDA、Attention、AttnRes、CausalConv 等；MoE 归类包括 GroupedMatmul、QuantBatchMatmul、量化/反量化、Moe routing/dispatch/combine、token permute、gating 等。其余归为 Other。`Total Time` 是算子累计时间，存在并行重叠，因此适合做负载构成和优化排序，不应当当作阶段串行时延。

## 1. P/D 阶段负载（Stage 资源视图）

|阶段|Stage (us)|计算 Computing (us / Stage)|通信总量 (us / Stage)|其中未重叠通信|通信重叠部分|计算引擎内访存|计算引擎内计算|
|---|---:|---:|---:|---:|---:|---:|---:|
|P|10,929,388|6,475,332 / **59.2%**|7,503,983 / **68.7%**|3,993,408 / **36.5%**|3,510,574 / **32.1%**|**51.3%**|**48.7%**|
|D|2,135,553|1,909,769 / **89.4%**|323,909 / **15.2%**|142,754 / **6.7%**|181,156 / **8.5%**|**50.4%**|**49.6%**|

P 的主要问题是通信：通信总量约为计算时间的 1.16 倍，且 36.5% 的阶段时间是未被计算覆盖的通信。D 的通信压力明显小，瓶颈转为计算内核本身；D 的计算占阶段 89.4%。两阶段的计算内核都没有显示出纯计算或纯访存一边倒的情况，MTE 与计算子计数器都约占活跃引擎时间的一半，说明算子融合、数据布局和量化路径都可能影响实际吞吐。

### Kernel 资源视图：访存 / 计算 / 通信

以下比例按 `kernel_details.csv` 的资源忙碌量归一化：MTE 计数器归为访存，MAC/标量/向量计数器归为计算，通信核时长归为通信。它用于比较 kernel 资源构成，不等同于 Stage 墙钟时间；资源之间可能并行重叠。

|阶段|访存|计算|通信|资源视图读法|
|---|---:|---:|---:|---|
|P Prefill|**26.8%**|**25.5%**|**47.7%**|通信占比最高，计算与访存接近|
|D Decode|**47.4%**|**46.6%**|**6.0%**|访存与计算接近，通信占比较低|

Stage 构成来自阶段墙钟时间线，Kernel 资源视图来自 kernel 内部计数器和通信核时长；两者统计层级不同，不能直接相加或互换。

## 2. Attention/MoE 负载构成

|阶段|算子累计时间 (us)|Attention (us / 阶段算子时间)|MoE (us / 阶段算子时间)|Other|
|---|---:|---:|---:|---:|
|P|7,985,320|2,005,161 / **25.1%**|4,043,506 / **50.6%**|24.3%|
|D|2,420,072|483,203 / **20.0%**|1,159,345 / **47.9%**|32.1%|

若只看 Attention+MoE，两者合计占 P 的 75.7%、D 的 67.9%。P/D 的 MoE 都比 Attention 占用更多算子时间；P 的 Attention 绝对负载更高，D 的 Attention 绝对时间下降后，MLA/KDA 的单 token 路径成为主要关注点。

### P 阶段

- Attention 内部：`ChunkKdaFwd` **66.5%**（占全体算子 16.7%），`FusedInferAttentionScore` **24.3%**（全体 6.1%），两者合计占 Attention **90.8%**。
- MoE 内部：`GroupedMatmul` **34.5%**（全体 17.5%），`DequantSituQuant` **19.4%**，`QuantBatchMatmulV3` **18.6%**，`DynamicQuant` **16.5%**；前四项合计占 MoE **89.0%**。

### D 阶段

- Attention 内部：`FlashMlaWithKvcache` **51.5%**（全体 10.3%），`RecurrentKda` **14.7%**，`AttnResFwdFused` **13.4%**，`MlaPrologV3` **7.9%**；前四项合计占 Attention **87.5%**。
- MoE 内部：`GroupedMatmulSituQuant` **35.0%**（全体 16.7%），`GroupedMatmul` **18.6%**，`QuantBatchMatmulV4` **14.3%**，`QuantBatchMatmulV3` **13.6%**，`MoeDistributeDispatchV3` **8.0%**，`MoeDistributeCombineV3` **6.3%**；前四项合计占 MoE **81.5%**，加上 dispatch/combine 后为 **95.8%**。

## 3. 瓶颈判断与优化优先级

### P Attention

首要瓶颈是 `ChunkKdaFwd`，其次是 `FusedInferAttentionScore`。建议优先检查 KDA 的 kernel 融合、KV/中间张量读写次数、布局转换和序列维度上的 tiling；随后检查 attention score 内核是否受 MTE2 读带宽、workspace 搬运或同步影响。P 阶段还要把 Attention 的通信等待和计算重叠起来，否则单独优化 Attention kernel 会被通信收益抵消。

### P MoE

P 的 MoE 是最大算子负载（50.6%）。优化顺序建议为 `GroupedMatmul` → `DequantSituQuant`/`DynamicQuant` → `QuantBatchMatmulV3`，再处理 `MoeTokenPermute/Unpermute` 和 routing。重点是 grouped matmul 的专家 token 聚合与小矩阵利用率、量化/反量化与 matmul 的融合、以及 token permute 的搬运开销。P 的 all-to-all/all-gather/reduce-scatter 通信也需要和 dispatch/combine、专家计算做流水重叠。

### D Attention

首要瓶颈是 `FlashMlaWithKvcache`，其后是 `RecurrentKda` 和 `AttnResFwdFused`。建议先看 KV cache 访问合并、cache layout、单 token 下的向量化和 MTE 利用率；`MlaPrologV3` 与 metadata 路径可通过减少临时张量和融合预处理降低固定开销。D 阶段通信比例低，attention kernel 本身的访存效率和 launch/同步开销更值得优先优化。

### D MoE

首要瓶颈是 `GroupedMatmulSituQuant` 和 `GroupedMatmul`，随后是 `QuantBatchMatmulV4/V3`。建议把 grouped matmul、situ quant、dequant/scale 计算放入更少的融合 kernel，针对 decode 的小 token batch 选择低延迟 tiling；同时检查 dispatch/combine 的数据搬运和专家负载均衡。D 的 MoE 算子累计时间占 47.9%，但阶段计算占 89.4%，因此应优先压缩 kernel 计算/访存时间，而不是先做通信优化。

## 4. 结论

P 阶段首先优化通信与 MoE：通信未重叠部分是最大的阶段级损失，算子级优先级为 `GroupedMatmul`、量化链路、`ChunkKdaFwd`。D 阶段首先优化计算 kernel：优先 `FlashMlaWithKvcache`、`GroupedMatmulSituQuant`/`GroupedMatmul` 和 QuantBatchMatmul 链路。两阶段都应以 MTE 和计算子计数器为依据做 kernel 融合、布局和 tiling 调优，并用端到端 Stage 时间复测，避免只看算子累计时间。
