# FORK 说明 — 6TBWhite/astrbot_plugin_continuous_message

本 fork 基于上游 [aliveriver/astrbot_plugin_continuous_message](https://github.com/aliveriver/astrbot_plugin_continuous_message) v2.5.0，
增加了**自适应防抖（adaptive debounce）**功能。

> 合入上游 PR 后此文件可删除。

---

## 新增功能

上游只有固定 `debounce_time` 防抖，本 fork 新增了一套不依赖 `input_status` 的自适应防抖机制，根据消息长度、结尾标点和连续短句动态调整等待时间，适合 SnowLuma 等暂不支持输入状态通知的 OneBot v11 实现。

### 新增配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_adaptive_debounce` | bool | `true` | 启用自适应防抖 |
| `adaptive_min_wait` | float | `1.0` | 最短等待时间（秒） |
| `adaptive_max_wait` | float | `6.0` | 最长等待时间（秒） |
| `adaptive_max_total_wait` | float | `12.0` | 单轮最长总等待时间（秒），新增消息不会刷新此上限 |
| `adaptive_short_message_threshold` | int | `10` | 短消息字数阈值，连续短句会适当延长等待 |

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

- 很短的消息、逗号/省略号结尾、连续短句：倾向于多等一会儿
- 较长的消息、问号/句号/感叹号结尾：倾向于更快结算
- 无结尾标点按消息长度轻量加权，视为聊天口语输入
- 单轮总等待不会超过 `adaptive_max_total_wait`

### 后续迭代方向

PR #16 已提交并包含了当前全部自适应防抖逻辑。后续可考虑：
- 根据实际使用反馈微调各档位的等待时间和加权值
