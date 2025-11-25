# main.py - 兼容图片识别和分段回复的消息防抖插件
import asyncio
from typing import List, Tuple
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig, logger
from astrbot.core.utils.session_waiter import session_waiter, SessionController


@register(
    "continuous_message",
    "aliveriver",
    "将用户短时间内发送的多条私聊消息合并成一条发送给LLM(仅私聊模式)",
    "1.1.0"
)
class ContinuousMessagePlugin(Star):
    """
    改进版消息防抖插件(仅私聊模式)
    - 只合并纯文字消息,图片消息跳过防抖
    - 合并后修改 event 内容,让消息走正常 LLM 流程
    - 兼容图片识别路由和分段回复插件
    """

    _ImageComponent = None
    _PlainComponent = None
    _image_component_import_failed = False
    
    try:
        from astrbot.api.message import Image as _ImageComponent
        from astrbot.api.message import Plain as _PlainComponent
    except ImportError:
        _image_component_import_failed = True
        logger.warning("[消息防抖动] 无法导入消息组件类,将使用类名检查作为后备方案")

    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}

        self.debounce_time = float(self.config.get('debounce_time', 2.0))
        self.command_prefixes = self.config.get('command_prefixes', ['/'])
        self.enable_plugin = self.config.get('enable', True)
        self.merge_separator = self.config.get('merge_separator', '\n')

        logger.info(f"[消息防抖动] 插件已加载 - 启用: {self.enable_plugin}, 防抖: {self.debounce_time}秒")

    def is_command(self, message: str) -> bool:
        """判断是否为命令消息"""
        message = (message or "").strip()
        if not message:
            return False
        for prefix in self.command_prefixes:
            if message.startswith(prefix):
                return True
        return False

    def _parse_message(self, message_obj) -> Tuple[str, bool]:
        """
        解析消息内容
        返回: (纯文本内容, 是否包含图片)
        """
        text = ""
        has_image = False

        try:
            components = getattr(message_obj, "message", None)
            if components is None:
                if hasattr(message_obj, "text"):
                    return (message_obj.text or "", False)
                return ("", False)

            for component in components:
                comp_class_name = getattr(getattr(component, "__class__", None), "__name__", "")
                
                # 提取文本
                if comp_class_name in ('Plain', 'Text'):
                    if hasattr(component, 'text'):
                        text += (component.text or "")
                    elif hasattr(component, 'content'):
                        text += (component.content or "")

                # 检测图片
                if self._ImageComponent is not None:
                    is_image = isinstance(component, self._ImageComponent)
                else:
                    is_image = comp_class_name == 'Image'

                if is_image:
                    has_image = True

        except Exception as e:
            logger.warning(f"[消息防抖动] 解析消息组件时出错: {e}")

        return text.strip(), has_image

    def _should_skip_message(self, event: AstrMessageEvent) -> Tuple[bool, str, bool]:
        """
        判断消息是否应该跳过防抖
        返回: (是否跳过, 文本内容, 是否有图片)
        """
        text, has_image = self._parse_message(event.message_obj)

        if not text:
            text = (event.message_str or "").strip()

        # 跳过条件: 有图片、空消息、或是命令
        skip = has_image or (not text) or (text and self.is_command(text))

        return skip, text, has_image

    def _modify_event_message(self, event: AstrMessageEvent, merged_text: str):
        """
        修改事件的消息内容为合并后的文本
        这样消息可以走正常的 LLM 流程
        """
        try:
            # 更新 message_str
            event.message_str = merged_text
            
            # 更新 message 组件
            if hasattr(event.message_obj, "message"):
                # 创建新的 Plain 组件
                if self._PlainComponent is not None:
                    new_plain = self._PlainComponent(text=merged_text)
                else:
                    # 后备方案:尝试复制第一个组件的结构
                    components = event.message_obj.message
                    if components and len(components) > 0:
                        first_comp = components[0]
                        comp_class = type(first_comp)
                        try:
                            new_plain = comp_class(text=merged_text)
                        except:
                            # 如果失败,直接修改文本属性
                            logger.warning("[消息防抖动] 无法创建新组件,直接修改现有组件")
                            if hasattr(first_comp, 'text'):
                                first_comp.text = merged_text
                            return
                    else:
                        logger.warning("[消息防抖动] 无法修改消息组件")
                        return
                
                # 替换为单个文本组件
                event.message_obj.message = [new_plain]
                
                logger.info(f"[消息防抖动] 已修改事件消息内容: {merged_text[:50]}...")
                
        except Exception as e:
            logger.error(f"[消息防抖动] 修改事件消息失败: {e}", exc_info=True)

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=50)
    async def handle_private_msg(self, event: AstrMessageEvent):
        """
        私聊消息防抖逻辑
        - 不阻断接收阶段事件链(不调用 stop_event)
        - 合并完成后修改 event 内容,让后续插件和 LLM 流程正常处理
        """
        if not self.enable_plugin:
            return

        # 检查是否应该跳过
        skip, raw_text, has_image = self._should_skip_message(event)
        if skip:
            return

        display_msg = raw_text[:50] if raw_text else ""
        logger.info(f"[消息防抖动] 开始防抖处理: {display_msg}")

        if self.debounce_time <= 0:
            return

        # 消息缓冲区
        buffer: List[str] = [raw_text]

        @session_waiter(timeout=self.debounce_time, record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            nonlocal buffer

            text, has_image = self._parse_message(ev.message_obj)
            if not text:
                text = (ev.message_str or "").strip()
            else:
                text = text.strip()

            # 如果收到图片或命令,停止收集
            if has_image:
                logger.info(f"[消息防抖动] 收到图片消息,停止防抖")
                controller.stop()
                return
            
            if text and self.is_command(text):
                logger.info(f"[消息防抖动] 收到命令消息,停止防抖")
                controller.stop()
                return

            # 跳过空消息
            if not text:
                return

            # 跳过重复的第一条消息
            if len(buffer) == 1 and text == buffer[0]:
                logger.info(f"[消息防抖动] 跳过重复处理的第一条消息")
                controller.keep(timeout=self.debounce_time, reset_timeout=True)
                return

            # 加入缓冲区
            buffer.append(text)
            logger.info(f"[消息防抖动] 收集消息 ({len(buffer)}): {text[:50]}")
            
            # 继续等待
            controller.keep(timeout=self.debounce_time, reset_timeout=True)

        try:
            await collect_messages(event)
            
            # 被 controller.stop() 停止(收到图片或命令)
            logger.info(f"[消息防抖动] 防抖被中断,已收集 {len(buffer)} 条消息")
            
            # 如果只收集到1条,不需要修改
            if len(buffer) == 1:
                return
            
            # 合并消息并修改 event
            merged_message = self.merge_separator.join(buffer).strip()
            if merged_message:
                logger.info(f"[消息防抖动] 合并消息: {merged_message[:100]}...")
                self._modify_event_message(event, merged_message)
            
            return

        except TimeoutError:
            # 防抖超时,合并消息
            merged_message = self.merge_separator.join(buffer).strip()
            if not merged_message:
                return

            logger.info(f"[消息防抖动] 防抖超时,合并了 {len(buffer)} 条消息")
            
            # 修改 event 的消息内容
            self._modify_event_message(event, merged_message)
            
            # 不阻断事件,让消息走正常流程
            return

        except Exception as e:
            logger.error(f"[消息防抖动] 插件内部错误: {e}", exc_info=True)
            return
