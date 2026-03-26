import asyncio
from typing import Dict
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig, logger

from .message_parser import MessageParser, IS_AIOCQHTTP
from .forward_handler import ForwardHandler
from .link_parser_adapter import LinkParserAdapter

# 检查是否为 aiocqhttp 平台
if IS_AIOCQHTTP:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

@register(
    "continuous_message",
    "aliveriver",
    "将用户短时间内发送的多条私聊消息合并成一条发送给LLM（仅私聊模式，支持合并转发消息、引用消息、输入状态感知）",
    "2.2.1"
)
class ContinuousMessagePlugin(Star):
    """
    消息防抖动插件 v2.2.1
    消息防抖动插件（仅私聊模式）
    
    功能：
    1. 拦截用户短时间内发送的多条私聊消息
    2. 在防抖时间结束后，将这些消息合并成一条发送给LLM
    3. 过滤指令消息，不参与合并
    4. 保持人格设定和对话历史
    5. 支持图片识别和传递
    6. 支持QQ合并转发消息的提取和合并（aiocqhttp平台）
    7. 支持QQ引用消息的智能识别和上下文标注（aiocqhttp平台）
    8. 支持输入状态感知，检测到用户正在打字时自动延长等待（NapCat等支持input_status的平台）

    安全设计：
    - 强制仅在私聊启用，避免群聊中不同用户的消息被误合并
    """
    
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}
        
        self.debounce_time = float(self.config.get('debounce_time', 2.0))
        self.command_prefixes = self.config.get('command_prefixes', ['/'])
        self.enable_plugin = self.config.get('enable', True)
        self.merge_separator = self.config.get('merge_separator', '\n')
        self.enable_forward_analysis = self.config.get('enable_forward_analysis', True)
        self.forward_prefix = self.config.get('forward_prefix', '【合并转发内容】\n')
        self.enable_typing_detection = self.config.get('enable_typing_detection', True)
        self.max_typing_wait = float(self.config.get('max_typing_wait', 5.0))

        # 引用消息配置
        reply_format = '[引用消息({sender_name}: {full_text})]'
        bot_reply_hint = self.config.get('bot_reply_hint', '[系统提示：以上引用的消息是你(助手)之前发送的内容，不是用户说的话]')
        
        # 会话存储
        self.sessions: Dict[str, Dict] = {}
        
        # 初始化子模块
        image_comp = None
        plain_comp = None
        try:
            from astrbot.api.message_components import Image, Plain
            image_comp = Image
            plain_comp = Plain
        except ImportError:
            try:
                from astrbot.api.message import Image, Plain
                image_comp = Image
                plain_comp = Plain
            except ImportError:
                logger.error("[消息防抖动] 严重: 组件导入失败")

        self.parser = MessageParser(image_component=image_comp, plain_component=plain_comp)
        self.forward_handler = ForwardHandler(reply_format=reply_format, bot_reply_hint=bot_reply_hint)
        self.link_parser = LinkParserAdapter(self.config)

        logger.info(
            f"[消息防抖动] v2.2.1 加载 | 事件驱动模式 | 防抖: {self.debounce_time}s "
            f"| 合并消息: {self.enable_forward_analysis} | 输入感知: {self.enable_typing_detection} "
            f"| 链接解析: {self.link_parser.enabled}"
        )

    async def terminate(self):
        await self.link_parser.close()

    async def _timer_coroutine(self, uid: str, duration: float):
        """
        计时器协程：等待指定时间后触发结算
        
        当有新消息到来时，旧计时器会被取消（CancelledError），
        新消息会创建新的计时器重新开始倒计时。
        """
        try:
            await asyncio.sleep(duration)
            if uid in self.sessions:
                self.sessions[uid]['flush_event'].set()
        except asyncio.CancelledError:
            pass

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=50)
    async def handle_private_msg(self, event: AstrMessageEvent):
        if not self.enable_plugin or self.debounce_time <= 0:
            return

        # 0a. 输入状态检测：根据 status_text 判断用户是否正在输入
        if self.enable_typing_detection and self.parser.is_typing_event(event):
            raw = event.message_obj.raw_message
            status_text = raw.get('status_text', '')
            event_type = raw.get('event_type', '')
            user_id = raw.get('user_id', '')
            uid = event.unified_msg_origin
            has_session = uid in self.sessions
            is_typing = '正在输入' in status_text
            logger.debug(f"[消息防抖动] 输入状态通知 | user_id: {user_id} | status_text: {status_text} | is_typing: {is_typing} | 有活跃会话: {has_session} | event_type: {event_type}")
            if has_session:
                session = self.sessions[uid]
                if is_typing:
                    # 正在输入：取消原计时器，启动超时保护计时器防止卡死
                    session['is_typing'] = True
                    if session.get('timer_task'):
                        session['timer_task'].cancel()
                    session['timer_task'] = asyncio.create_task(
                        self._timer_coroutine(uid, self.max_typing_wait)
                    )
                    logger.info(f"[消息防抖动] 用户正在输入，暂停结算（超时保护 {self.max_typing_wait}s） - 用户: {uid}")
                else:
                    # 停止输入：仅在之前确实处于输入状态时才恢复防抖倒计时
                    # 避免重复的 is_typing=False 通知反复重置计时器
                    if session.get('is_typing'):
                        session['is_typing'] = False
                        if session.get('timer_task'):
                            session['timer_task'].cancel()
                        session['timer_task'] = asyncio.create_task(
                            self._timer_coroutine(uid, self.debounce_time)
                        )
                        logger.info(f"[消息防抖动] 用户停止输入，恢复防抖 {self.debounce_time}s - 用户: {uid}")
                    else:
                        logger.debug(f"[消息防抖动] 忽略重复的停止输入通知 - 用户: {uid}")
            event.stop_event()
            return

        # 0. 检测并处理合并转发消息（仅aiocqhttp平台）
        forward_text = ""
        forward_images = []
        if self.enable_forward_analysis and IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent):
            forward_id = await self.forward_handler.detect_forward_message(event)
            if forward_id:
                try:
                    forward_text, forward_images = await self.forward_handler.extract_forward_content(event, forward_id)
                    if forward_text or forward_images:
                        logger.info(f"[消息防抖动] 检测到合并转发 | 文本: {len(forward_text)}字 | 图片: {len(forward_images)}张")
                except Exception as e:
                    logger.error(f"[消息防抖动] 提取合并转发失败: {e}")
            else:
                reply_text, reply_images = await self.forward_handler.extract_reply_content(event)
                if reply_text or reply_images:
                    forward_text = reply_text
                    forward_images = reply_images

        # 1. 解析消息内容
        raw_text, has_image, current_urls = self.parser.parse_message(event.message_obj)
        if not raw_text:
            raw_text = (event.message_str or "").strip()
        
        # 合并转发内容处理
        if forward_text:
            if forward_text.startswith('[引用消息('):
                raw_text = forward_text + ("\n" + raw_text if raw_text else "")
            else:
                prefix_text = self.forward_prefix + forward_text
                raw_text = prefix_text + ("\n" + raw_text if raw_text else "")
        if forward_images:
            current_urls.extend(forward_images)
            has_image = True
        
        uid = event.unified_msg_origin

        # 2. 处理指令消息：立即中断当前防抖会话并结算
        if self.parser.is_command(raw_text, self.command_prefixes):
            if uid in self.sessions:
                if self.sessions[uid].get('timer_task'):
                    self.sessions[uid]['timer_task'].cancel()
                self.sessions[uid]['flush_event'].set()
            return

        # 3. 忽略空消息
        if not raw_text and not has_image:
            return

        # ================== 核心防抖逻辑 ==================

        # 场景 A: 追加到现有会话 (Msg 2, 3...)
        if uid in self.sessions:
            session = self.sessions[uid]
            
            if raw_text:
                session['buffer'].append(raw_text)
            if current_urls:
                session['images'].extend(current_urls)
            
            # 重置计时器
            if session.get('timer_task'):
                session['timer_task'].cancel()
            
            session['timer_task'] = asyncio.create_task(
                self._timer_coroutine(uid, self.debounce_time)
            )
            
            event.stop_event()
            return

        # 场景 B: 启动新会话 (Msg 1)
        flush_event = asyncio.Event()
        timer_task = asyncio.create_task(
            self._timer_coroutine(uid, self.debounce_time)
        )
        
        self.sessions[uid] = {
            'buffer': [raw_text] if raw_text else [],
            'images': current_urls,
            'flush_event': flush_event,
            'timer_task': timer_task,
            'is_typing': False
        }
        
        logger.info(f"[消息防抖动] 开始收集 - 用户: {uid}")

        await flush_event.wait()
        
        # ================== 结算阶段 ==================
        if uid not in self.sessions:
            return
        session_data = self.sessions.pop(uid)
        
        buffer = session_data['buffer']
        all_images = session_data['images']
        merged_text = self.merge_separator.join(buffer).strip()
        merged_text, all_images = await self.link_parser.enrich(merged_text, all_images)
        
        if not merged_text and not all_images:
            return

        img_info = f" + {len(all_images)}图" if all_images else ""
        logger.info(f"[消息防抖动] 结算触发 - 共 {len(buffer)} 条{img_info} -> 发送")
        logger.info(f"[消息防抖动] 合并后的完整消息:\n{merged_text}")
        if all_images:
            logger.debug(f"[消息防抖动] 图片列表: {all_images}")
        
        self.parser.reconstruct_event(event, merged_text, all_images)
        return
