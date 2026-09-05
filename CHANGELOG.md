# 更新日志

## v2.9.1

### 问题修复
- 🐛 **修复 weixin_oc 等无持久图片 URL 平台，防抖窗口内第 2 条及之后消息的图片丢失**
  - 现象：私聊先发一张图再补发文字/图片时，后续消息中的图片无法被 LLM 识别，控制台报 `provider.entities:216: 图片预处理结果为空，将忽略。`
  - 原因：AstrBot v4 会把 temp 目录下的入站媒体登记为事件级临时文件，在该事件 pipeline 结束时删除；防抖窗口内第 2+ 条消息的事件先于结算结束，其图片文件被框架清理，结算重构的 Image 组件指向已删除文件。#17/#21 的修复（优先 raw_message 持久 URL）仅覆盖图片自带 http URL 的 aiocqhttp 类平台，微信 CDN 图片无持久引用可提取
  - 方案：消息进入防抖队列前调用 `MessageParser.preserve_images()`，把本地临时图片立即读取并固化为 `base64://` 引用；读不到的图片直接丢弃并记录警告
  - 资源限制：固化前检查单张图片大小与单条消息图片总量（`preserve_image_max_bytes` 默认 20MB、`preserve_images_total_max_bytes` 默认 50MB，≤0 不限制），超限记录警告并跳过，防止大图/多图造成较高内存占用
  - 远程图片（http/https）、data URI、base64 引用不受影响，原样保留

### 技术改进
- 新增 `MessageParser.preserve_images()` / `_is_local_media_ref()` 方法（文件读取在独立线程执行）
- 新增配置项 `preserve_image_max_bytes`、`preserve_images_total_max_bytes`
- 结算阶段 debug 日志对 base64 引用截断展示

---

## v2.9.0

### 新增

- 新增 `image_vision` 配置组，可为私聊图片指定独立的 VLM Provider。
- 专用 VLM 识别成功后将图片描述交给主会话模型，避免 AstrBot 因主模型支持多模态而跳过配置的 VLM。
- 专用 VLM 调用失败时保留原有图片并回退 AstrBot 默认处理流程。

## v2.8.1

### 修复

- 放宽 Pillow 版本约束至 `>=10.0.0,<13.0.0`，兼容 AstrBot Core v4.27.3 锁定的 Pillow 12.2.0，避免插件更新时触发核心依赖版本保护冲突。

## v2.8.0

### 新增

- 新增 ID 黑/白名单配置组 `access_control`，可控制哪些用户的消息参与防抖合并及其处理方式。
- 黑名单与白名单可独立启用，支持按发送者 ID、会话 ID 或完整会话来源标识匹配；黑名单优先级高于白名单，名单为空时不过滤任何用户。
- 黑白名单逻辑封装为独立模块 `access_control.py`，主流程只保留调用点。
- 黑名单改为 `blacklist_mode` 三档下拉：`disable` 不限制、`immediate` 不合并等待收到后立即处理（其余功能照常）、`skip` 完全跳过本插件处理。

## v2.7.0

### 新增

- 新增可选图片本地化配置组 `image_handling`，可将远程图片 URL 下载到本地缓存后通过 `Image.fromFileSystem` 交回 AstrBot。
- 新增本地化图片缓存自动清理，可按 `image_localization_cleanup_max_age_hours` 删除过期文件。
- 新增 GIF 第一帧转 JPG 选项，降低不同 VLM/provider 对动图支持不一致造成的问题。

### 改进

- 图片本地化逻辑拆分到 `image_localizer.py`，减少主流程代码体积。

### 兼容性

- GIF 转 JPG 依赖 `Pillow`，若运行环境缺少该依赖，会保留原 GIF 并继续后续流程。

## v2.6.0

### 新增

- 新增自适应防抖策略，可根据消息长度、结尾标点和连续短句数量动态调整等待时间。
- 新增单轮最长总等待时间，避免用户持续补充消息时无限延后结算。
- 配置文件改为 AstrBot v4.26 推荐的嵌套分组写法：`basic`、`debounce`、`message_features`、`qq_card`、`image_handling`、`link_parser`。

### 改进

- 适配 AstrBot v4.26 插件元数据写法，插件信息改由 `metadata.yaml` 声明。
- 保留旧版扁平配置兼容，插件启动时会自动把嵌套配置展开为内部旧键名。
- 事件监听改为 `EventMessageType.ALL` 后再由插件内部判断私聊，以便接收 aiocqhttp 的撤回、输入状态等非普通私聊消息事件。

### 修复

- 修复 AstrBot v4.26 中调用 `event.stop_event()` 会留下空 `MessageEventResult`，导致 RespondStage 出现空内容发送的问题。
- 插件内部吞掉中间消息时改用静默终止逻辑：停止传播、清空 result，并阻止默认 LLM 请求。

### 兼容性

- 最低 AstrBot 版本调整为 `>=4.26.0`。
- 图片是否能被模型理解仍取决于 AstrBot 中配置的模型是否支持视觉输入。

## v2.5.0

### 改进

- 引用消息与合并转发内容统一改为 XML 标签包裹，降低 LLM 误解析风险。
- 引用消息格式改为 `<quoted_message sender="...">...</quoted_message>`。
- 合并转发格式改为 `<forward_content>...</forward_content>`。
- 移除 `bot_reply_hint` 配置项，改由 `sender` 属性表达引用来源。
- `forward_prefix` 保留为兼容字段，默认不再额外添加前缀。
- 防抖核心流程补充更详细的 `debug` 日志。

## v2.4.0

### 新增

- 新增撤回消息过滤。
- 防抖等待期间，用户撤回的消息会从待合并队列中移除。
- 若所有待合并消息均被撤回，本轮结算会终止，不向 LLM 发送空内容。

### 技术改进

- 会话数据新增 `items` 列表，记录每条消息的 `message_id`、文本与图片。
- 新增撤回事件识别和撤回消息 ID 提取逻辑。

## v2.3.0

### 新增

- 新增 QQ 卡片链接提取。
- 新增链接解析增强，可为支持的平台补充标题、正文、封面、作者等信息。
- QQ 卡片解析支持：B 站、小红书、小黑盒、百度贴吧、NGA、网易云音乐、知乎。
- 链接解析支持：B 站、小红书、小黑盒、网易云音乐。

### 配置

- 新增 `enable_qq_card_parsing`、`qq_card_disabled_platforms`、`qq_card_prompt`。
- 新增 `enable_link_parsing`、`link_parser_disabled_platforms`、`link_parser_success_prompt`、`link_parser_failure_prompt`。
- 新增 `link_parser_merge_images`、`link_parser_max_links`、`link_parser_timeout`、`link_parser_max_text_length`、`link_parser_proxy`。

## v2.2.1

### 修复

- 修复重复的停止输入通知导致防抖计时器被反复重置的问题。
- `max_typing_wait` 默认值调整为 `60` 秒，更适合长时间输入场景。

## v2.2.0

### 新增

- 新增输入状态感知，检测到用户正在输入时暂停结算。
- 停止输入后恢复防抖倒计时。
- 增加超时保护，避免平台不发送停止输入事件导致会话卡住。

### 重构

- 拆分为模块化结构：`main.py`、`message_parser.py`、`forward_handler.py`。

## v2.1.1

### 新增

- 新增 QQ 引用消息识别。
- 自动提取被引用消息的文本和图片上下文。
- 支持识别引用消息是否来自 Bot 自身。

### 修复

- 修复引用消息内容重复显示的问题。
- 修复 Bot 自身消息引用提示不稳定的问题。
- 修复引用消息与用户消息之间缺少分隔的问题。

## v2.1.0

### 新增

- 新增 QQ 合并转发消息提取。
- 支持用户直接发送合并转发消息。
- 支持用户回复或引用合并转发消息。
- 合并转发中的文本和图片会纳入防抖流程。

## v2.0.0

### 重大更新

- 重构为事件驱动架构。
- 使用 `asyncio.Event` 挂起等待，降低空转开销。
- 使用 `asyncio.Task.cancel()` 实现精确计时器重置。
- 通过重构消息事件与 AstrBot 后续流程兼容。

## v1.0.0

### 初始版本

- 基于会话等待机制实现私聊消息防抖。
- 支持文本合并、图片透传、指令过滤和基础配置。
