

> 基于 vllm-ascend @ `528276dc9`（2026-08-05）与 vLLM @ `d0ce3dadb` 源码整理，行号对应两仓当前工作区。

> 姊妹文档：`kv_cache_management_detailed.md`（vLLM 主干的 KV 管理，本文的对照基准）；

> 前置阅读：`vllm_ascend_patch_timeline.md`、`vllm_ascend_platform_patches.md`、`vllm_ascend_worker_patches.md`。

---

## 目录

1. vllm-ascend 作为 OOT 组件的工作原理（极简版）
2. KV 管理总体方案：主干复用 + 冲突点 patch + worker 侧重写
3. 两张流程图：EngineCore 与 Worker 的功能动作
4. 调度侧（engine-core）的 KV 改动5. worker 侧（NPU）的物理分配与布局6. 模型分章：GQA / MLA / SFA / DSA(DSv4) / KDA / GLM5.2 与 vLLM 的差异（调度管理/物理格式/PD/prefix/投机五维）7. 对照总表与设计原则

---

# 1. vllm-ascend 作为 OOT 组件的工作原理（极简版）

vllm-ascend **不改 vLLM 源码**，靠「平台插件 + monkey patch + 正式注册接口」挂载（细节见三份 patch 文档）：

```text

主进程:  import vllm → 平台发现 NPUPlatform          【接缝1：platform 插件】

        → pre_register_and_update → ★ A 组 platform 补丁（配置定型前打）

EngineCore: load_general_plugins（幂等）→ ★ B 组注册（connector/loader/model）

        → 调度器已被 A 组补丁改过（KV coordinator / balance schedule / kv_cache_utils）

Worker:   NPUWorker.__init__ → ★ C 组 worker 补丁（模型加载前打）

        → register_ascend_customop：26 个算子 OOT  【接缝5：CustomOp.__new__】

        → attention 后端正式注册（AttentionBackendEnum.CUSTOM, "ASCEND"）【接缝6】

        → PluggableLayer OOT 替换模型层（KDA/GDN）   【接缝7】

```

与 KV 管理直接相关的挂载点：

| 挂载点 | 内容 | 域 |

|---|---|---|

| `patch/platform/patch_kv_cache_utils.py` | 替换 5 个 KV 规划函数 | 调度侧 |

| `patch/platform/patch_kv_cache_coordinator.py` | `get_kv_cache_coordinator` → Ascend 版 | 调度侧 |

| `patch/platform/patch_mamba_manager.py` | `AscendMambaManager` | 调度侧 |

| `patch/platform/patch_mamba_config[_310].py` | mamba/attention page 对齐 | 配置期 |

| `patch/worker/patch_v2/patch_attn_utils.py` | v2 runner 的分配/reshape/spec | worker |

| `patch/worker/patch_v2/patch_block_table.py` | `AscendBlockTables`（int32） | worker |

| `patch/worker/patch_deepseek_v2.py` | MLA init 跳过 shared indexer | worker |

| `patch/worker/patch_qwen3_5.py` | GDN 线性注意力 | worker |

| `patch/worker/patch_mamba_utils.py` | mamba state 拷贝 | worker |

A/C 分组的判据是**时机不是进程**：platform 组必须在配置定型前打（KV 规划在那时执行），worker 组可以等到 `NPUWorker.__init__` 再打。

---

# 2. KV 管理总体方案

## 2.1 一句话原则

**调度侧主干（BlockPool、prefix cache、请求状态机、三种布局框架）原样复用上游；与 NPU 冲突的规划点用 platform 补丁改写；worker 侧的物理 tensor 分配/视图完全由 Ascend 自写**（v1 runner）或整体替换（v2 runner patch）。

对照 vLLM 主干的三层（`kv_cache_management_detailed.md` §1）：

| 层 | vLLM（CUDA） | vllm-ascend（NPU） |

|---|---|---|

| 调度入口（Scheduler） | 上游 | 上游 + `patch_balance_schedule`（DP 均衡，与本题无关不展开） |

| 分配层（coordinator/manager） | 上游 | 上游 + 3 个 KV patch（§4） |

| 物理层元数据（BlockPool） | 上游 | **上游原样**（`AscendHybridKVCacheCoordinator.__init__` 里自建同款 BlockPool，patch_kv_cache_coordinator.py:118-124） |

| worker 物理 tensor | `_allocate_kv_cache`（attn_utils.py:183） | **完全自写**（model_runner_v1.py:3902，§5） |

两条 runner 路径：

- **v1 runner（默认）**：`NPUModelRunner`（`vllm_ascend/worker/model_runner_v1.py`），KV 分配/视图自写，不走上游 attn_utils；
- **v2 runner（`VLLM_USE_V2_MODEL_RUNNER`）**：复用上游 `vllm/v1/worker/gpu/` runner，靠 `patch_v2/patch_attn_utils.py` 把上游 `_allocate_kv_cache`/`_reshape_kv_cache`/`bind_kv_cache`/`get_kv_cache_spec` 整体换成 `vllm_ascend/worker/v2/attn_utils.py` 的等价实现（patch 落点 `patch_v2/patch_attn_utils.py:17-21`）。

---

# 3. 两张流程图：EngineCore 与 Worker 的功能动作

按「谁、在什么时机、做什么功能动作」组织初始化和分配两个流程；黄色节点 = 该动作的代码被 patch/自写、走 Ascend 实现，白色 = 复用上游实现。图后附「动作 → 实现方 → 目的」速查表。

## 3.1 初始化流程（`_initialize_kv_caches` 的调用层级，A.B 表示父调子）

黄色 = 该动作的代码被 patch / 自写，走 Ascend 实现；白色 = 复用上游实现。

`_initialize_kv_caches`（core.py:250）本体的编号为 A，其子调用为 A.1、A.2…；worker 侧编号为 B。

## 3.2 分配流程（每个调度 step，`Scheduler.schedule` / `execute_model` 的调用层级）

编号规则同 §3.1：C = 调度侧（`Scheduler.schedule` 及其子调用，C.2.x = `allocate_slots` 内部步骤），D = worker 侧（`execute_model` 及其子调用）；①-⑩ = 功能动作序号，与 §3.3 速查表对应。黄色 = 该动作的代码被 patch/自写，白色 = 复用上游。

## 3.3 功能动作 → 实现方 → 目的 速查

**初始化（对应 §3.1 的 A.x / B.x 调用层级）**

| 动作 | 实现位置 | 目的 | 详见 |

|---|---|---|---|

| A.1 注册 spec→manager | ① patch_mamba_manager.py:88（偷换模块名，MambaSpec→AscendMambaManager）② platform.py:156 register_custom_kv_cache_specs → register_ascend_kv_cache_specs（kv_cache_interface.py:224-239，注册 AscendMLAAttentionSpec / AscendSFAIndexerCacheSpec / AscendSlidingWindowMLASpec） | 调度侧 manager 选择对 NPU 透明 | §4.5 |

| A.2 收集 spec | v1：NPUModelRunner.get_kv_cache_spec（:4686，自有类原生方法）；v2：patch_v2/patch_attn_utils.py:20 | 自有 spec 承载 NPU 特有记账（compress_ratio/scale_dim/indexer） | §6.3-6.4 |

| A.3 显存 profiling | 上游 determine_available_memory | — | — |

| A.4.1 分组 | get_kv_cache_groups 本体上游；patch_kv_cache_utils.py:249-250 替换其 DSv4 分支（:59-182） | DSv4 按 compress_ratio 拆 C4/C128 组；普通模型分组与上游一致 | §4.2 |

| A.4.3 布局+num_blocks | get_kv_cache_config_from_groups 本体上游；patch_kv_cache_utils.py:254 替换 _get_kv_cache_config_packed（:185-245） | NPU tensor 需连续独立分配，不用 packed | §4.3 |

| A.5 调度侧 config | 上游 | — | — |

| A.6 广播+worker 分配 | 上游 RPC；worker 侧见 B.x | — | — |

| B（块大小解析） | patch_kv_cache_utils.py:248 → _ascend_resolve_kv_cache_block_sizes（:23-56） | 允许多组+DCP（MLA/SWA-MLA 各自独立 CP） | §4.1 |

| S（Scheduler 实例化） | patch_kv_cache_coordinator.py:390-456 | DSv4 强制 AscendHybrid coordinator | §4.4 |

| B.1 connector 初始化 | 上游（worker.py:978 同款顺序） | — | — |

| B.2.1 initialize_kv_cache_tensors | model_runner_v1.py:3731 | 内部分两步：B.2.1.1 分配 + B.2.1.2 view | §5.1 |

| B.2.1.1 分配显存 | model_runner_v1.py:3902-4108 | K/V 分离 + 2MB 对齐，为 PD 分离传输服务 | §5.2 |

| B.2.1.2 view 成形 | model_runner_v1.py:4143-4487 | NPU 布局：K/V 分 tensor、num_blocks-first、as_strided | §5.3-5.4 |

| B.2.2 绑定+注册 | model_runner_v1.py:3716-3720 | connector 按 (k,v) tuple 解包注册 | §5.6 |

**分配（对应 §3.2 的 ①-⑩）**

| 动作 | 实现位置 | 目的 | 详见 |

|---|---|---|---|

| ① prefix 查找 | patch_kv_cache_coordinator.py:266-387 | DSv4 有 C4/C128 两个 full 组，命中需各自截断 | §4.4 |

| ② 计算块数 | patch_mamba_manager.py:50-85 | 外部 cache 同步加载时容量检查准确性 | §4.5 |

| ③④⑤ 检查/取块/写回 | 上游 | 账本逻辑跨平台通用，零改动 | — |

| ⑥ block table | v2/block_table.py:51-61 / worker/block_table.py | NPU reshape_and_cache 算子要求 int32 | §5.5 |

| ⑦ slot_mapping | ops/triton/compute_slot_mapping.py | NPU triton kernel 差异 | §5.5 |

| ⑧ forward | vllm_ascend/attention/* | NPU 算子（正式注册接口，非 patch） | §6 各章 |

| ⑨ post_forward | connector 全家桶（kv_p2p/kv_pool） | (k,v) 二元组、SFA c8 合并注册 | §5.6 |

| ⑩ 回收 | 上游 | — | — |

读图要点：

1. **patch 集中在「规划」和「物理」两端**：中间的账本（BlockPool、prefix cache 哈希、LRU、请求状态机——分配的 ③④⑤⑩）一行没改；
2. **初始化改的是「怎么规划」，alloc 改的是「怎么算数和怎么搬运」**——分配的 ①② 是规划差异在运行时的投影，⑥-⑨ 全是物理差异；
3. 看完这两张图再读 §4（调度侧详解）和 §5（worker 侧详解），每个 patch 的"为什么"都能落到图上的具体动作。

---

# 4. 调度侧（engine-core）的 KV 改动

## 4.1 块大小解析：`patch_kv_cache_utils.py:23-56`

上游 `resolve_kv_cache_block_sizes`（vllm/v1/core/kv_cache_utils.py:606-668）：多组时**只对 AttentionSpec 乘 dcp，Mamba 组不乘**，再 LCM；hash 粒度取 GCD 并带一串回退/校验。

Ascend 版（`_ascend_resolve_kv_cache_block_sizes`）：

```python

group_block_sizes = [g.kv_cache_spec.block_size for g in groups]

scheduler_block_size = math.lcm(*group_block_sizes) * dcp      # :50 全部组（含 Mamba）先 lcm 再乘 dcp

if not cache_config.enable_prefix_caching:

    return scheduler_block_size, scheduler_block_size

hash_block_size = math.gcd(*group_block_sizes)                 # :53 原始块大小的 gcd（粒度更细）

```

动机（patch 注释 :29-36）：上游 PR#40860 给「多 block size 混合 + DCP」加了限制（`HybridKVCacheCoordinator.__init__` 断言 DCP 下只允许 FullAttention/Mamba 组，vllm/v1/core/kv_cache_coordinator.py:571-580），该限制对 CUDA 成立，但 **Ascend 对 MLA 和 SWA-MLA 各自独立实现了 context parallelism**，可以跑 DCP + 多组。补丁同时覆盖 `vllm.v1.engine.core` 里被 `from import` 缓存的旧引用（patch_kv_cache_utils.py:256-259）。

## 4.2 DSv4 分组：`group_and_unify_kv_cache_specs`（:59-90）

| | 上游（kv_cache_utils.py:1558） | Ascend |

|---|---|---|

| full MLA 分组 | 全部合并成 1 个 UniformType 组 | **按 `compress_ratio` 分组**（:74-75），C4 组排最前（:78） |

| SWA 分组键 | `(block_size, sliding_window)` | 只按 `block_size`（:73） |

| 返回 | `[full_mla, *swa]` | **`[C4 MLA, C128 MLA, *swa]`**（:182） |

`_get_kv_cache_groups_uniform_groups`（:93-182）与上游共享同一套 `_approximate_gcd` 对齐算法（选 num_layer_tuples 使 padding 最小），但 **Ascend 独有「SWA 层 page 向上 padding」**（:139-157）：把每个 SWA 层的 `page_size_padded` 抬到 full-MLA 桶集合中最近的不小于它的桶大小——因为 Ascend 不用 packed 布局（见 §4.3），非 packed 要求同 tensor 槽位各层 page 一致。

## 4.3 物理配置：非 packed 的 `_get_kv_cache_config_deepseek_v4`（:185-245）

上游 DSv4 走 **packed 布局**（一整块 backing tensor + strided view，kv_cache_utils.py:1296）。**Ascend 整体替换为非 packed**：

```python

# 每个 (layer_tuple 序号, page_size 桶) 发一个独立 KVCacheTensor

KVCacheTensor(size=ps * num_blocks, shared_by=...)   # :234-241

# 没有 offset / block_stride —— 各 tensor 是独立物理分配，大小各异

num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)  # :230

# MTP 层不分桶，各自单独一个 tensor（:209-221, :242-243）

```

前提（docstring :192-196）：SWA 层已在 §4.2 做过 page padding，每层 page 都落在 full-MLA 桶集合上。**对照结论：vLLM 的三种布局（单组/槽位/packed）在 Ascend 上只有前两种 + 这个 DSv4 专用的「按桶独立 tensor」变体，没有 packed。**

## 4.4 `AscendHybridKVCacheCoordinator`（patch_kv_cache_coordinator.py:76-387）

工厂（:390-456）：DSv4 配置（`model_version == "deepseek_v4"` 探测，:47-64）→ 强制 Ascend 版；单组/无缓存 → 回落上游；其余多组 → Ascend 版。

与上游 `HybridKVCacheCoordinator` 的差异：

1. **`__init__` 不调 super()**，手工重建字段（:76-177），并**删除上游 DCP 断言**（上游 kv_cache_coordinator.py:571-580）——这是 DCP+SWA-MLA 能跑通的另一半；
2. **有效块大小 ×dcp ×compress_ratio**（`_get_effective_block_size`，:185-195；上游直接用 `manager.block_size`）；
3. 写路径对齐：把 `SlidingWindowManager.scheduler_block_size` 设为 `lcm_block_size`（:173-175），使 reachable mask 粒度与命中粒度一致；
4. `find_longest_cache_hit`（:266-387）主体同上游不动点算法，但命中后**对所有 full-attention 组各自截断**（:374-384，DSv4 有 C4/C128 两个 full 组），上游只截 `attention_groups[0]`；
5. `enable_partial_hash_hits` 专门处理 mamba align 模式（:160-165）。

## 4.5 `AscendMambaManager`（patch_mamba_manager.py:20-88）

整体替换上游 `MambaManager`，实质差异只有一处（其余透传/继承）：

```python

# get_num_blocks_to_allocate（:50-85）：先调上游实现，然后：

has_external_tokens = total_computed_tokens > local_hit_tokens

has_new_scheduled_tokens = num_tokens_main_model > total_computed_tokens

if has_external_tokens and has_new_scheduled_tokens:

    num_new_blocks += 1        # :82-84 外部 cache 与新 token 同步加载时多算一块

```

原因（:73-79 注释）：AscendStoreConnector 的外部 KV 与新 token 同步加载时 `allocate_new_computed_blocks()` 会额外分配一块，不补则空闲容量检查算错。

---

# 5. worker 侧（NPU）的物理分配与布局

## 5.1 v1 runner 分配调用链

```text

NPUWorker.initialize_from_config()              vllm_ascend/worker/worker.py:976-987

  ├─ ensure_kv_transfer_initialized()           （connector 先初始化，与上游同序）

  ├─ CaMemAllocator.use_memory_pool(tag="kv_cache")   （sleep 模式内存池）

  └─ model_runner.initialize_kv_cache(config)   model_runner_v1.py:3662-3723

       ├─ apply_layerwise_kv_cache_plan()       （层间 KV 复用规划，:3674）

       ├─ initialize_attn_backend()             （建 attn_groups，:3677）

       ├─ initialize_kv_cache_tensors()         :3731

       │    ├─ _allocate_kv_cache_tensors()     :3902-4108  ← int8 裸分配

       │    └─ _reshape_kv_cache_tensors()      :4143-4487  ← view 出形状

       └─ register_kv_caches                    :3716-3720（connector / sparse offload）

```

## 5.2 核心差异一：K/V 物理分离的二元组

**vLLM（CUDA）**：普通 attention 每层一个 tensor（K/V 打包，FA 是 `(B,H,N,2D)` 单 tensor + stride_order）。

**vllm-ascend**：普通 attention 每层返回 **`(k_tensor, v_tensor)` 两个独立物理 tensor**（model_runner_v1.py:4033-4099），K/V 按维度比例瓜分 page 预算：

```python

# model_runner_v1.py:4061-4062

k_tensor_size = kv_cache_tensor.size // calc_split_factor([k_dim, v_dim])

# MLA 时 k_dim/v_dim = kv_lora_rank(512) / qk_rope_head_dim(64)

# → K=nope cache，V=rope cache，两个独立 tensor

```

动机（:3907-3908 注释）："To support prefill disaggregation, we need to split kvcache tensor into k_cache and v cache, and the addr of both are aligned by 2M"——**P/D 分离要求 K、V 各自 2MB 对齐**（`alignment = 2*1024*1024`，:3921；`kv_transfer_config` 非空时才多分配并对齐切片，:3956-3961）。

reshape 阶段（:4263-4449）：K、V 各自 view 成 `(num_blocks, block_size, num_kv_heads, head_size)`，最终 `kv_caches[layer_name] = (k_cache, v_cache)`（:4448-4449）。

## 5.3 核心差异二：布局约定

| 维度 | vLLM（CUDA） | vllm-ascend |

|---|---|---|

| K/V 存储 | 单 tensor 打包（FA content 维或 (2,...)） | **物理分离两个 tensor** |

| 维度顺序 | 依赖 `get_kv_cache_stride_order` + permute | **固定 num_blocks-first，无 stride_order**（后端 shape 自带 4 元组，无后端重写 stride_order） |

| KV cache NZ 格式 | 无 | **也不用**——NZ 只出现在 mooncake 传输路径（mooncake_layerwise_connector.py:1847）和权重上；A5 上 attention impl 内部另建 NZ 5D 视图（attention_v1.py:956-958），与存储布局无关 |

| slot_mapping dtype | int64 | **int32**（NPU reshape_and_cache 算子要求，v2/block_table.py:51-61；v1 block_table.py:102-105） |

| hybrid page 对齐 | 槽位布局 + `page_size_padded` | 共享 buffer 三表交错 + `as_strided` 跳 padding（:4468-4473，见 §5.4） |

| packed 布局 | DSv4 默认 | **不支持**（§4.3） |

## 5.4 各 spec 的物理分支（`_allocate/_reshape` 内）

- **Mamba / linear_attn / HiddenStateCache**：单个整 tensor（:3937-3979；mamba 与 hidden-state 必要时分两份物理内存）；
- **SFA indexer**：`(k_tensor,)` 或 `(k_tensor, scale_tensor)`；scale 版从**同一块对齐裸内存**切两段，减少 HCCL/Mooncake 注册条数（:3847-3900，注释 :3866-3868）；
- **hybrid（attention+mamba 共享 page）**：三张表交错（:4468-4473 注释）——`tensor1: [(kv_padding), conv, ...]`、`tensor2: [k, ssm, ...]`、`tensor3: [v, (mamba_padding), ...]`，attention 的 K/V 视图落在共享 buffer 尾部；
- **DSv4/压缩 KV**：`_adjust_kv_layout`（:4110-4140）用 `torch.as_strided` 按 `page_size_bytes // dtype_size` 的块步长建多段视图（A5 还有 third full-KV overlap 视图）；
- **`page_size_padded`**：`as_strided` 视图跳过页内填充（:4305-4342）。

## 5.5 BlockTable / InputBatch 差异

- **`AscendBlockTables`**（v2/block_table.py:23-61）：slot_mapping 重建为 **int32**（上游 int64）；
- **v1 runner 的 `BlockTable`/`MultiGroupBlockTable`**（worker/block_table.py:16-441）：int32、DCP 下 `duplicate_size` 扩行、hybrid 物理→逻辑块展开（`_convert_physical_to_logical_blocks` :295-310）、Mamba 组跳过 slot mapping（:435-441）、compress_ratio 缩减 `max_num_blocks_per_req`（:33-49）、slot mapping 用 Ascend 自研 triton kernel（ops/triton/compute_slot_mapping.py）；
- **`NPUInputBatch`**（worker/npu_input_batch.py:38）：不调 `super().__init__`，自重建全部 buffer。

## 5.6 KV connector 注册侧

- 注册时机与上游一致（initialize_kv_cache 末尾，model_runner_v1.py:3719-3720），但 dict 的 value 是 **`(k_cache, v_cache)` 二元组**（或 mamba state list）——Ascend 自带 connector 全部按 tuple 解包逐 tensor 注册（如 mooncake_layerwise_connector.py:1278-1303）；
- Ascend 的 connector 全家桶在 `vllm_ascend/distributed/kv_transfer/`（mooncake × 3、sfa_pd_rd2h、ascend_store、ucm、recompute_cpu_offload、sparse_kv_offload）；
- 分配侧配合：P/D 时 2MB 对齐（§5.2）；SFA c8 的 k+scale 合并分配减少注册条数（§5.4）。

---

# 6. 模型分章：与 vLLM 的 KV 管理差异

每章按五个维度展开：**调度管理**（spec/分组/coordinator/manager/块大小）、**物理 tensor 格式**（布局/shape/dtype/对齐）、**PD 分离**、**prefix cache**、**投机解码**。

## 6.1 GQA（Qwen/Llama/GLM-4.5/4.6 等稠密模型）

**调度管理**

- 零改动基准：`FullAttentionSpec` 透传（model_runner_v1.py:4722-4725），单组 → 工厂直接回落上游 `UnitaryKVCacheCoordinator`（patch_kv_cache_coordinator.py:426-440）；
- 唯一差异：kernel block size 固定 **128**（attention_v1.py:148-149），`refresh_block_size` 把默认 `cache_config.block_size` 抬到 128（utils.py:1185-1197）；310P 可选 [128, 64]（_310p:98-99）。

**物理 tensor 格式**

- 与 vLLM 的单 tensor K/V 打包（FA `(B,H,N,2D)` + stride_order）不同：**K/V 物理分离两个 tensor**（§5.2），shape `(2, num_blocks, block_size, heads, head_size)`（attention_v1.py:110-117），impl 内 `kv_cache[0]/kv_cache[1]` 取用（:1294）；
- 310P 特殊 5D 对齐布局 `(2, num_blocks, (heads*dim)//16, block_size, 16)`（_310p/attention/attention_v1.py:56-81）；A5 在 impl 内部另建 NZ 5D 视图（attention_v1.py:956-958，不影响存储布局）。

**PD 分离**：走 mooncake 系 connector（kv_p2p/mooncake_connector.py:1567 等）；K/V 分离 + 2MB 对齐的物理设计主要就是为它服务（model_runner_v1.py:3907-3908）；注册时 `(k,v)` 二元组逐 tensor 注册。

**prefix cache**：完全上游机制（链式 hash + FullAttentionManager），无 Ascend 特有改动。

**投机解码**：支持。Eagle3 有两个补丁——`patch_eagle3_init`（修上游 PP 下 `target_layer_num` 算错的 bug，:129 行补丁注释）、`patch_draft_quarot`（QuaRot 旋转权重加载，条件生效）；v2 路径另有 `patch_v2/patch_eagle_speculator`（EagleAclGraphManager）。

## 6.2 MLA（DeepSeek V2/V3、Kimi K2 系）

**调度管理**

- `AscendMLABackend`（"ASCEND_MLA"，mla_v1.py:72）+ `AscendMLAAttentionSpec`（vllm_ascend/core/kv_cache_interface.py:19，比上游多 `compress_ratio`/`scale_dim`/`storage_block_size` 字段供 DSv4 复用）；
- 单组稠密 MLA，调度路径与上游一致；**DCP 独立实现**（context_parallel/mla_cp.py:162）——配合 §4.1/§4.4 删除的上游 DCP 断言，MLA+DCP 在 Ascend 可用、在 CUDA 上游被禁；
- kernel block size 128（mla_v1.py:106-107）。

**物理 tensor 格式**

- 上游单 tensor `(num_blocks, bs, 576)` → Ascend **拆 nope/rope 两个 tensor**：`(num_blocks, bs, 1, 512)` + `(num_blocks, bs, 1, 64)`（model_runner_v1.py:4405-4428，按 512:64 比例瓜分 page）；
- backend shape 为 4 元组 `(num_blocks, bs, 1, 576)`（mla_v1.py:88-95，多一维恒 1 的 kv-head 维；上游 MLACommon 是 3 元组）；A5 走 FP8。

**PD 分离**：mooncake / mooncake_layerwise（kv_p2p/，layerwise 传输时才把 tensor 转 NZ，mooncake_layerwise_connector.py:1847）；K/V(nope/rope) 分离 + 2MB 对齐直接服务 PD。

**prefix cache**：机制同上游；差异在 `patch_mla_prefill_backend`——上游会调 `get_mla_prefill_backend()` 检查可用性，Ascend 替换为「`is_available()=True`、调用即报错」的哨兵（真正 prefill 在 `AscendMLAImpl` 内部完成）。

**投机解码**：`deepseek_mtp` 支持，`patch_deepseek_mtp`（:80 行）修 MTP 权重名解析（`rot.weight`）并替换 4 个 MTP 类；Eagle3 也有 DeepSeek 变体支持（`Eagle3DeepseekV2ForCausalLM`）。

## 6.3 SFA（DeepSeek V3.2、GLM-5.1 的稀疏注意力）

**调度管理**

- 两类 spec 同组：`AscendMLAAttentionSpec`（主 cache）+ **`AscendSFAIndexerCacheSpec`**（indexer K/scale，kv_cache_interface.py:107-168：block_size=128、head_size=128、可选 scale_dim=1）；
- 与上游的关键差异：indexer 不用真后端，用**占位后端 `AscendSFAIndexerBackend`**（indexer.py:14-30，`get_impl_cls()=None`），并**注册 `FullAttentionManager` 从而与主 cache 共享 block id**（kv_cache_interface.py:230-234）——上游是把 indexer 当独立 `MLAAttentionSpec` 层走 UniformType 分组（详见主文档 §7.2）；
- DCP 下 indexer 按 `sfa_dcp_replicated_indexer_size` 全量复制（model_runner_v1.py:424-426, :4242）。

**物理 tensor 格式**

- kv_cache tuple（sfa_v1.py:564-582 的表）：默认 `(k_nope, k_pe, indexer_k, indexer_scale)`；**sparse C8** 时 `(packed_kv, indexer_k, indexer_scale)`——nope(int8/fp8)+pe+scale 按 `SFA_QSFA_TILE_SIZE` tile 打包成单 tensor（:1369-1384）；
- indexer 的 k+scale 从同一块对齐裸内存切两段，减少 HCCL/Mooncake 注册条数（model_runner_v1.py:3866-3868）。

**PD 分离**

- **专用 connector `sfa_pd_rd2h`**（kv_p2p/sfa_pd_rd2h/connector.py 文件头）：D 侧把**主 MLA KV 直接拉进 CPU pinned offload 池**，**indexer KV 落 HBM**——稀疏注意力只需 indexer 在显存里选 top-k，主 KV 可放 CPU 按需取；CPU 池和稀疏常驻 cache 由 `SparseKVOffloadManager` 管理；
- 也可走 mooncake 系；稀疏 KV offload 变体把主 cache 注册为 6-tuple（sfa_kv_offload.py:63/136）。

**稀疏 KV offload**（Ascend 独有特性，上游无对应；`ascend_config.py:1257-1299` 校验 + `sparse_kv_offload_manager.py:302`）

- 语义：D 节点主 MLA KV 进 CPU pinned 池、仅 indexer K/scale 留 HBM，靠稀疏性用 top-k 按需取主 KV；
- 门槛：仅 SFA 系（有 `index_topk` 且无 `compress_ratios`，DSv4 显式拒绝）；禁 CP/PP/v2 runner；默认仅 PD 分离 D 节点（`keep_device_kv_cache` 仅供调试）；与 sparse SFA C8 互斥；
- 对 worker 分配的影响：B.2.1.1 独立分支 `allocate_kv_cache_tensors_for_sparse_kv_offload`（model_runner_v1.py:4063-4077）、B.2.1.2 对应 reshape（:4272-4284）、注册侧 `store_on_host`（:4752）。

**prefix cache**：全量常驻语义与上游一致（top-k 可选任意历史 token，KV 不能部分 evict）；indexer 与主 cache 同 block id、同 hash，命中/驱逐天然成对（同上游 DSA）。

**投机解码**：`deepseek_mtp` 支持；GLM-5.1 的 IndexCache 机制（运行时复用别层 top-k，权重仍每层有）与 GLM-5.2 的 shared indexer（见 §6.6）要区分。

## 6.4 DSA（DeepSeek-V4，压缩 KV 系——Ascend 改动最深）

**调度管理**

- **分组重写**（§4.2）：full MLA 按 compress_ratio 拆 **C4 / C128 两个组**（上游是一组混排 layer tuple），SWA 只按 block_size 分组；SWA page 向上 padding 到 full-MLA 桶；
- **coordinator 重写**（§4.4）：`AscendHybridKVCacheCoordinator`——删上游 DCP 断言（DCP+SWA-MLA 可跑）、有效块大小 ×dcp ×compress_ratio（:185-195）、双 full 组命中截断（:374-384）、`SlidingWindowManager.scheduler_block_size = lcm`；
- 压缩换算由 `AscendMLAAttentionSpec.storage_block_size` + **`CompressAttentionManager`**（kv_cache_interface.py:224-229）承载；
- block_size 限 **32/64/128，默认压到 32**（utils.py:1202-1211）；子 cache 物理块尺寸表 `DSV4_BLOCK_SIZES`（models/layer/attention/layer.py:30-48）。

**物理 tensor 格式**

- **非 packed**（§4.3）：按 (layer_tuple, page 桶) 独立 tensor + MTP 层单独 tensor；`_adjust_kv_layout` 用 `as_strided` 建多段视图（model_runner_v1.py:4110-4140，A5 有 third full-KV overlap）；
- indexer 是 `AscendDeepseekV4IndexerCache`（models/deepseek_v4/indexer.py:83-120），**直接复用 AscendDSABackend**，spec 是 `AscendMLAAttentionSpec(scale_dim=1)`；`AscendDSAImpl` 一层持有 indexer+compressor+swa_cache 三类子层（dsa_v1.py:873）；
- A5 走 FP8（dsa_v1.py:999）。

**PD 分离**：mooncake_hybrid_connector（kv_p2p/mooncake_hybrid_connector.py:1081，hybrid 多组场景）；SWA 层滑出窗口的块在 `request_finished` 前先 `remove_skipped_blocks` 不传给 D（沿用上游 scheduler.py:2589-2595 的机制）。

**prefix cache**：`enable_partial_hash_hits`（patch_kv_cache_coordinator.py:160-165）处理细粒度命中；双 full 组的命中截断是 Ascend 特有；eagle 组回退（:127-130）。

**投机解码**：**DSpark**（`DSparkDraftModel` 注册，models/deepseek_v4/dspark.py）；`patch_speculative_config` 为 DSpark 补 `ptd_token_id`（从 `dspark_noise_token_id`/`mask_token_id` 回退）；MTP 由 `DeepSeekV4MTPModel` 注册（models/deepseek_v4/mtp.py）。

## 6.5 KDA（Kimi K3 / KimiLinear；含 Qwen3-Next GDN 对照）

**调度管理**

- KDA 层 → 上游 `MambaSpec`（state shape/dtype 完全继承，未重写）；MLA 层 → `AscendMLAAttentionSpec`；多组走 `AscendHybridKVCacheCoordinator`（`enable_partial_hash_hits` 专门处理 mamba align，:160-165）；
- **state 管理三层 patch**：`AscendMambaManager`（外部 cache 同步加载 `num_new_blocks += 1`，§4.5）+ `patch_mamba_config[_310]`（page 对齐，见下）+ `patch_mamba_utils`（拷贝路径，见 prefix cache）；
- page 对齐（`patch_mamba_config`）：`attn_block_size = 128 × cdiv(ssm_page, 128 × attn_k_page)` 且**断言严格相等**（:94-97），mamba page 补到 `attn_page + conv_page`（:113-117）；用 AscendStoreConnector 时**强制 `mamba_cache_mode="align"`**（:136-146），align 时 `mamba_block_size = attention block_size`（:143-146）。310P 版：最小 128 倍数使 `attn_page ≥ mamba_page`（:75-83），pad 到精确相等（:92-99），"all" 模式按 `lcm(chunk_size, 128)` 对齐（:68-73）。

**物理 tensor 格式**

- 层接入走 **OOT PluggableLayer 替换**（utils.py:737），非 patch；模型本体加载上游 amd 分支（models/kimi_k3.py:17-33）；
- hybrid 共享 buffer 三表交错（§5.4），attention K/V 视图落 buffer 尾部；310P ssm state 强制 fp16（_310p/ops/fla/gdn_310.py:149-151）；
- kernel：自研 Triton `chunk_kda`/`fused_recurrent_kda` + AscendC `npu_causal_conv1d_custom`（kimi_kda.py:107-243）；GDN 另有 AscendC `npu_recurrent_gated_delta_rule`（自研算子，gdn.py:457）。

**PD 分离**：用 **AscendStoreConnector**（kv_pool/ascend_store）；mamba state 的 PD 依赖 align 模式（强制开启）+ 延迟拷贝——`patch_mamba_utils` 的 `preprocess_mamba` **只收集拷贝元数据、不在此执行拷贝**，实际拷贝推迟到 KV transfer load 之后（:226-296，:292-293 注释）；310P 不能跑 Triton 指针拷贝，走纯 tensor 拷贝回退（:66-185，专门处理 padded page 的不连续 state 视图）。

**prefix cache**：align 模式 checkpoint 机制同上游（`i × block_size` 边界缓存 state）；拷贝实现差异——Triton `batch_memcpy_kernel` 的 `BLOCK_SIZE` 从 1024 改 **8192**（:26-34，NPU 性能）、`MambaCopyBuffers` 的 uint64 指针 buffer 改 int64（:207-223，NPU aclnnInplaceZero 不支持 DT_UINT64）。

**投机解码**：**KDA 不支持**（kimi_kda.py:151-154，遇 `spec_sequence_masks` 直接 `raise NotImplementedError`；也不支持 DS conv state 布局，:165-169）。GDN（Qwen3-Next）支持（gdn.py:294-326 的 spec 分支）。

## 6.6 GLM-5.2（`glm_moe_dsa`）

**调度管理**

- 架构 = DeepSeek V2 子类（上游 deepseek_v2.py:1936），走 **SFA 全套**（§6.3 的 spec/分组/coordinator）；
- 核心增量：**shared indexer**——GLM-5.2 checkpoint 在共享层直接不存 Indexer 权重（`indexer_types[layer_id] == "shared"`），`patch_deepseek_v2._should_skip_indexer_init`（:36-54, :229-255）只对「skip_topk 且标记 shared」的层跳过 Indexer 构造，这些层不产 indexer spec；MTP 层始终保留完整 Indexer（:46-47）；
- 与 GLM-5.1 的区分：5.1 只是运行时 IndexCache 复用 top-k（每层仍有完整 Indexer 权重），**不跳过构造**——混同会导致 5.1 权重加载断（patch/**init**.py:583-593）。

**物理 tensor 格式**：同 SFA tuple；shared 层无 indexer tensor，forward 时必须配 `topk_indices_buffer` 复用别层算好的 top-k（sfa_v1.py:505-516）；GLM 的 indexer RoPE 风格特判（`is_rope_neox_style=False` + torch_npu lightning indexer，sfa_v1.py:531-534）。

**PD 分离**：教程推荐 **MooncakeConnectorV1**（docs/source/tutorials/models/GLM5.2.md）；部署形态还有 DSA-CP（attention/context_parallel/dsa_cp.py）、sparse LI C8。

**prefix cache**：同 SFA（全量常驻）；shared indexer 层不影响 hash 语义（top-k 索引是计算的确定性产物，只需主 KV 命中即可重算）。

**投机解码**：`deepseek_mtp` 支持，但 **GLM-5.2 不支持图模式投机，必须 `enforce_eager`**（GLM5.2.md）；e2e 有 GLM-5.2-w4a8 + DSpark/MTP 投机测试（tests/e2e/pull_request/eight_card/test_glm5_2.py:33-36）。

# 7. 对照总表与设计原则

## 7.1 一页对照

| 维度 | vLLM（CUDA） | vllm-ascend（NPU） |

|---|---|---|

| 调度侧主干 | BlockPool/coordinator/prefix cache | **原样复用**，冲突点 3 个 patch |

| DCP + 多组 | 禁止（PR#40860 断言） | **允许**（MLA/SWA-MLA 各自独立 CP） |

| DSv4 分组 | full MLA 一组（C4/C128 混排） | **C4/C128 拆两组** + SWA page padding |

| DSv4 布局 | packed（一块 backing + stride view） | **非 packed**（按桶独立 tensor） |

| 普通 attention 物理 | 单 tensor K/V 打包 | **K/V 分离二元组，P/D 时 2MB 对齐** |

| 默认 block size | 16 | **128**（DSv4 限 32/64/128） |

| slot_mapping | int64 | **int32** |

| KV NZ 格式 | 无 | 无（NZ 只在传输和权重） |

| spec 家族 | FullAttentionSpec/MLAAttentionSpec/MambaSpec/... | +**AscendMLAAttentionSpec / AscendSFAIndexerCacheSpec / AscendSlidingWindowMLASpec**（注册进上游 KVCacheSpecRegistry，kv_cache_interface.py:224-239） |

| indexer 管理 | 独立 MLAAttentionSpec 层，同组 | **自有 spec + 占位后端，显式共享 block id** |

| mamba/KDA | 上游 MambaManager | +3 层 patch（manager/config/utils） |

## 7.2 模型 × 特性速查

| 模型 | backend | 物理格式要点 | PD 分离 | prefix cache | 投机解码 |

|---|---|---|---|---|---|

| GQA | AscendAttentionBackend | K/V 分离两 tensor，块 128 | mooncake 系（K/V 分离+2MB 对齐就是为 PD） | 完全上游 | Eagle3（+PP bug 修复补丁） |

| MLA | AscendMLABackend | nope/rope 拆两 tensor，4 元组多一维 kv-head | mooncake / layerwise（传输时转 NZ） | 上游机制 + prefill 工厂哨兵 | deepseek_mtp（rot 权重补丁） |

| SFA | AscendSFABackend | tuple(k_nope,k_pe,idx_k,idx_scale)；C8 打包 | **sfa_pd_rd2h**：主 KV→CPU 池、indexer→HBM | 全量常驻，idx 与主 cache 共享 block id | deepseek_mtp |

| DSA(DSv4) | AscendDSABackend | 非 packed 按桶独立 tensor，C4/C128 两组 | mooncake_hybrid | partial hash hits + 双 full 组截断 | DSpark + MTP |

| KDA | AscendGDNAttentionBackend | MambaSpec 继承上游；hybrid 三表交错 | AscendStore（强制 align）+ 延迟拷贝 | align checkpoint；memcpy 8192/uint64→int64 | **不支持**（GDN 支持） |

| GLM5.2 | AscendSFABackend | 同 SFA；shared 层无 indexer tensor | MooncakeConnectorV1 + DSA-CP | 同 SFA；shared indexer 不影响 hash | deepseek_mtp，**必须 enforce_eager** |

## 7.3 设计原则（与 vLLM 主干对照后的提炼）

1. **账本与金库的边界在 OOT 下依然成立**：调度侧账本（BlockPool、状态机）跨平台通用，Ascend 一行没改；改的全是「规划算法」（分组/块大小/对齐）和「金库物理」（分配/视图/kernel）。
2. **patch 最小化原则**：能走正式接口的绝不 patch——attention backend 走 `AttentionBackendEnum.CUSTOM`、模型层走 `PluggableLayer`/CustomOp OOT、spec 走 `KVCacheSpecRegistry`；只有「没接口的规划函数」（kv_cache_utils/coordinator/manager）才 monkey patch。
3. **NPU 物理约束决定了所有差异的形态**：K/V 分离与 2MB 对齐（P/D 传输要求）、block_size=128（kernel 对齐）、int32 slot_mapping（算子 dtype）、无 packed（非连续视图不受支持/不需要）、310P 的 attn page ≥ mamba page（硬件连续性要求）。
4. **模型差异收敛到 spec + backend + 分组三件事**：GQA 零改动；MLA 换 spec 和物理切分；SFA 加占位后端与共享 block id 记账；DSv4 改分组+布局+coordinator 全家桶；KDA 走 OOT 层替换 + state 三层 patch；GLM5.2 = SFA + shared indexer 特判。

> 备注：`vllm_ascend/worker/v2/README.md:16-20` 仍把 `_allocate_kv_cache`/`_reshape_kv_cache` 列为 TODO，但实际已在 `v2/attn_utils.py:522/696` 实现并被 patch——README 已过时。`USE_MULTI_GROUPS_KV_CACHE`（patch_kv_cache_coordinator.py:33）在当前快照是无引用的死常量。