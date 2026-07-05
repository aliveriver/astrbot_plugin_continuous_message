# FORK 说明 — 6TBWhite/astrbot_plugin_continuous_message

本 fork 基于上游 [aliveriver/astrbot_plugin_continuous_message](https://github.com/aliveriver/astrbot_plugin_continuous_message)，
在 v2.5.0 基础上增加了自适应防抖（adaptive debounce）功能，并对算法做了进一步调优。

> 合入上游 PR 后此文件可删除。

---

## 与上游的差异

### 代码调优（`main.py`）

| 项目 | 本 fork | 上游 master |
|------|---------|-------------|
| 中等消息等待 (11-30字) | **2.7s** | 2.5s |
| 长消息等待 (31-80字) | **1.8s** | 1.5s |
| 句尾标点（。！!）减少等待 | **-0.5s** | -0.4s |
| 无结尾标点处理 | 按长度加权：≤10字 +0.8s / ≤30字 +0.7s / ≤80字 +0.3s | 固定 +1.0s |
| 无标点语义 | 视为"口语无标点"（chat_plain_end） | 视为"无结束标点"（no_end_punctuation） |

核心思路：无结尾标点的消息按**聊天口语输入**处理，根据消息长度轻量加权，不再一概视为"输入未完成"。

---

## 自适应防抖配置

需配合 `enable_adaptive_debounce: true` 使用（默认开启），适用于 SnowLuma 等不支持 `input_status` 的 OneBot v11 实现。

### enable_adaptive_debounce（启用自适应防抖）
- **类型**：布尔值
- **默认值**：`true`
- **说明**：根据消息长度、结尾标点和连续短句动态调整等待时间
- **补充**：新增消息只影响下一轮等待时间，不会刷新单轮总等待上限

### adaptive_min_wait（自适应最短等待时间）
- **类型**：浮点数（秒）
- **默认值**：`1.0`
- **说明**：动态等待的常规最短时间
- **补充**：接近单轮总等待上限时，实际等待可能短于该值

### adaptive_max_wait（自适应最长等待时间）
- **类型**：浮点数（秒）
- **默认值**：`6.0`
- **说明**：单次重置计时器时的最长等待时间

### adaptive_max_total_wait（单轮最长总等待时间）
- **类型**：浮点数（秒）
- **默认值**：`12.0`
- **说明**：从第一条消息进入缓冲开始，最长等待多久后强制结算
- **补充**：新增消息不会刷新这个总上限

### adaptive_short_message_threshold（短消息字数阈值）
- **类型**：整数
- **默认值**：`10`
- **说明**：小于等于该长度的消息会被认为更可能是分段表达，连续短句会适当延长等待

### SnowLuma 推荐配置

```json
{
  "debounce_time": 2.0,
  "enable_typing_detection": false,
  "enable_adaptive_debounce": true,
  "adaptive_min_wait": 1.0,
  "adaptive_max_wait": 6.0,
  "adaptive_max_total_wait": 12.0,
  "adaptive_short_message_threshold": 10
}
```

### 自适应防抖规则概览

- 很短的消息、逗号/省略号结尾、无标点口语短句、连续短句：倾向于多等一会儿
- 较长的消息、问号/句号/感叹号结尾：倾向于更快结算
- 无标点按聊天口语输入处理，根据消息长度轻量加权
- 单轮总等待不会超过 `adaptive_max_total_wait`，避免持续追加消息导致机器人一直不回复
