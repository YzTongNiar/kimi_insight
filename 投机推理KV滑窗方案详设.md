# vLLM-Ascend 草稿模型滑动窗口 Attention 设计方案

## 背景与目标

### 背景
在投机解码（Speculative Decoding）中，草稿模型（Draft Model）需要访问目标模型（Target Model）的 KV Cache 来生成候选 token。对于长上下文场景，草稿模型每次 Attention 都要扫描整个 KV Cache，带来大量不必要的计算开销。

### 目标
为 vLLM-Ascend 中的 EAGLE3 草稿模型增加**滑动窗口 Attention（Sliding Window Attention）**功能，限制草稿模型在 Attention 时只关注最近 `W` 个 token（W 可配置），从而：
- 降低计算量：Attention 复杂度从 O(L) 降至 O(W)，W << L
- 保持 KV 全量存储：仍保留完整 KV Cache 用于目标模型或其他用途，只改变草稿模型的可见范围

### 约束条件
- 不依赖特定 Attention Kernel 实现（如 FlashAttention 的 window_size 参数），采用通用方案
- 图模式兼容（内存地址固定）
- 对原有代码侵入最小，主要修改集中在 `llm_base_proposer.py`

---

## 方案设计

### 核心思想
- **窗口大小**：`W`（单位：token 数），由命令行参数 `--speculative-draft-window-size` 指定
- **块对齐**：由于 vLLM 使用 PagedAttention，KV Cache 按 block 组织，Attention 的可见范围必须对齐到 block 边界，以保证 block table 索引有效。设 block 大小为 `B`
- **固定窗口起始位置**：在一步投机解码中，草稿模型需要生成 `K` 个 token（`K = num_speculative_tokens`）。以最终长度 `L_final = L + K`（L 为当前序列长度）为基准计算窗口起始位置，使得在多步生成期间窗口左边界保持固定。这样只需在第一步构建一次窗口 block table，后续步直接复用
- **更新策略**：每步只更新 `seq_lens`（序列长度递增，但不超过 W），`block_table` 保持不变

### 关键公式
- 当前序列长度：`L`
- 草稿生成步数：`K`（`num_speculative_tokens`）
- 窗口大小：`W`
- 最终长度：`L_final = L + K`
- 窗口起始 token 索引：`start_token = max(0, L_final - W)`
- 对齐到 block 边界后的起始 token：`aligned_start = (start_token // B) * B`，其中 `B` 为 block 大小
- 起始 block 索引：`start_block = aligned_start // B`

不需要显式定义窗口 block 数量；实际使用时，从 `full_block_table` 的第 `start_block` 个 block 开始，连续读取足够覆盖 `W` 个 token 的 blocks 即可（block 数量为 `ceil(W / B)`，由代码内部计算）。由于 causal mask 的存在，即使读取的 blocks 中包含超出实际序列长度的部分，也不会 attend 到无效 token。

### 实现位置
主要修改文件：`vllm_ascend/spec_decode/llm_base_proposer.py`  
涉及类：`AscendSpecDecodeBaseProposer`  
方法：`__init__`、`_propose`

---

## 详细实现

### 参数配置

在 `__init__` 中增加窗口相关成员：

    class AscendSpecDecodeBaseProposer:
        def __init__(self, ...):
            # ... 原有初始化代码 ...
            self.draft_window_size = self.runner.speculative_config.draft_window_size  # None 表示禁用
            if self.draft_window_size is not None:
                self.block_size = self.runner.block_size   # KV cache block size
                # 窗口需要的 block 数量（实现用，非公式）
                self.window_blocks = (self.draft_window_size + self.block_size - 1) // self.block_size

### `_propose` 方法中的改动

**修改时机**：在 `_propose` 中，调用 `set_inputs_first_pass` 获得 `common_attn_metadata` 之后，以及后续对 metadata 进行二次修改（如 PCP/DCP 调整）之前，立即插入滑动窗口逻辑。

#### 滑动窗口逻辑封装

建议单独实现一个方法 `_apply_sliding_window`，在 `_propose` 中调用。以下给出该方法的具体实现：

    def _apply_sliding_window(self, common_attn_metadata, full_seq_lens):
        """应用滑动窗口，修改 common_attn_metadata 中的 block_table 和 seq_lens"""
        if self.draft_window_size is None:
            return
        full_block_table = common_attn_metadata.block_table_tensor
        num_reqs = full_seq_lens.shape[0]
        K = self.num_speculative_tokens
        W = self.draft_window_size
        B = self.block_size
        window_blocks = self.window_blocks
    
        # 最终长度 L+K
        final_seq_lens = full_seq_lens + K
        # 窗口起始 token
        start_tokens = (final_seq_lens - W).clamp(min=0)
        # 对齐到 block 边界
        start_blocks = (start_tokens // B) * B
        start_block_indices = (start_blocks // B).to(torch.int64)
    
        # 构建窗口 block table
        window_block_table = torch.empty(
            num_reqs, window_blocks,
            dtype=full_block_table.dtype, device=full_block_table.device
        )
        for i in range(num_reqs):
            start_idx = start_block_indices[i].item()
            end_idx = start_idx + window_blocks
            if end_idx <= full_block_table.shape[1]:
                window_block_table[i] = full_block_table[i, start_idx:end_idx]
            else:
                valid = full_block_table.shape[1] - start_idx
                window_block_table[i, :valid] = full_block_table[i, start_idx:]
                window_block_table[i, valid:] = 0   # 填充无效 block
    
        # 替换 common_attn_metadata 中的字段
        common_attn_metadata.block_table_tensor = window_block_table
        common_attn_metadata.seq_lens = torch.min(full_seq_lens, torch.full_like(full_seq_lens, W))
    
        # 预填充 seq_lens_group（供多步循环使用）
        for step in range(K):
            current_len = full_seq_lens + step
            windowed_len = torch.min(current_len, torch.full_like(current_len, W))
            self.seq_lens_group[step][:num_reqs].copy_(windowed_len)
            self.seq_lens_group[step][num_reqs:].fill_(0)

#### 在 `_propose` 中调用

在 `set_inputs_first_pass` 返回后，立即调用上述方法。注意需要先保存原始 `full_seq_lens`（因为调用后 `common_attn_metadata.seq_lens` 会被修改）。

    def _propose(self, ...):
        # ... 前面的代码 ...
    
        # 调用 set_inputs_first_pass 获得 common_attn_metadata
        num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = \
            self.set_inputs_first_pass(...)
    
        # 保存原始 seq_lens（未窗口化之前）
        full_seq_lens = common_attn_metadata.seq_lens.clone()  # 需要 clone 避免后续被修改
    
        # 应用滑动窗口（修改 common_attn_metadata 和 seq_lens_group）
        self._apply_sliding_window(common_attn_metadata, full_seq_lens)
    
        # 后续代码（如图模式调度、PCP/DCP 处理等）保持不变
        # 注意：如果有分支会覆盖 block_table_tensor（如 pcp/dcp），需添加条件判断：
        # if self.draft_window_size is None and ... :
        #     common_attn_metadata.block_table_tensor = ...
    
        # 多步循环不变
        for draft_step in range(self.num_speculative_tokens):
            common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
            common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]
            common_attn_metadata.seq_lens = self.seq_lens_group[draft_step][:num_reqs]
            common_attn_metadata.query_start_loc = self.query_start_loc_group[draft_step][: num_reqs + 1]
            # ... 调用模型 forward

#### 处理可能的冲突分支

原代码中可能存在以下覆盖 `block_table_tensor` 的片段（例如 PCP/DCP 模式）：

    if self.pcp_size * self.dcp_size > 1 and draft_step > 0:
        common_attn_metadata.block_table_tensor = self.block_table_tensor_clone[:slicing_length]

需要修改为：

    if self.draft_window_size is None and self.pcp_size * self.dcp_size > 1 and draft_step > 0:
        common_attn_metadata.block_table_tensor = self.block_table_tensor_clone[:slicing_length]

即仅在未启用滑动窗口时才允许覆盖。

---

## 边界情况与异常处理

| 场景                                | 处理方式                                                              |
| --------------------------------- | ----------------------------------------------------------------- |
| `L + K <= W`                      | `start_token = 0`，窗口覆盖全部历史，`windowed_len_t = L + t`，效果等同于无窗口。     |
| `L + K > W` 但 `L < W`             | 窗口起始为 0，但窗口大小仍为 W，`windowed_len_t` 逐渐增大直至 W 后保持不变。                |
| 请求的 block 表长度不足 `window_blocks`   | 填充 0 或 -1。建议填充 -1 并确保 kernel 能跳过无效 block（若 kernel 不支持，可维持 0 并测试）。 |
| 未启用窗口（`draft_window_size = None`） | 跳过所有新增逻辑，行为与原版一致。                                                 |
|                                   |                                                                   |


