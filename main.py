import asyncio
import json
import logging
from typing import Dict, List
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig
from astrbot.core.utils.session_waiter import session_waiter, SessionController

logger = logging.getLogger(__name__)


@register(
    "continuous_message",
    "aliveriver",
    "将用户短时间内发送的多条私聊消息合并成一条发送给LLM（仅私聊模式）",
    "1.0.0"
)
class ContinuousMessagePlugin(Star):
    """
    消息防抖动插件（仅私聊模式）
    
    功能：
    1. 拦截用户短时间内发送的多条私聊消息
    2. 在防抖时间结束后，将这些消息合并成一条发送给LLM
    3. 过滤指令消息，不参与合并
    4. 保持人格设定和对话历史
    5. 支持图片识别和传递
    
    安全设计：
    - 强制仅在私聊启用，避免群聊中不同用户的消息被误合并
    """
    
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}
        
        # 从配置读取参数
        self.debounce_time = float(self.config.get('debounce_time', 2.0))
        self.command_prefixes = self.config.get('command_prefixes', ['/'])
        self.enable_plugin = self.config.get('enable', True)
        self.merge_separator = self.config.get('merge_separator', '\n')
        
        # 输出到 logger
        logger.info(f"[消息防抖动] 插件已加载 - 启用: {self.enable_plugin}, 防抖: {self.debounce_time}秒")
    
    def is_command(self, message: str) -> bool:
        """
        检查消息是否是指令
        
        Args:
            message: 消息内容
            
        Returns:
            bool: 如果是指令返回True，否则返回False
        """
        message = message.strip()
        if not message:
            return False
            
        for prefix in self.command_prefixes:
            if message.startswith(prefix):
                return True
        return False
    
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=50)
    async def handle_private_msg(self, event: AstrMessageEvent):
        """
        私聊消息防抖逻辑（仅私聊可用，避免群聊越权问题）
        
        - 如果是指令消息：直接放行，不干预
        - 否则，参与防抖聚合：
          - debounce_time 秒内的新消息会被合并
          - 期间若出现指令，则结束本轮聚合
          - 超时后，把本轮聚合的文本一次性交给 LLM
        """
        # 如果插件未启用，直接返回
        if not self.enable_plugin:
            return
        
        message_str = (event.message_str or "").strip()
        
        # 如果消息为空或是指令，直接放行
        if not message_str or self.is_command(message_str):
            return
        
        logger.info(f"[消息防抖动] 开始防抖处理: {message_str[:50]}")
        
        # 防抖时间 <= 0，不进行防抖
        if self.debounce_time <= 0:
            return
        
        # 消息缓冲区（第一条消息也会通过 collect_messages 加入）
        buffer: List[str] = []
        
        # 存储图片 URL 列表
        image_urls = []
        
        # 会话控制器：收集消息 + 超时判断
        # 注意：第一条消息也会进入这个函数
        @session_waiter(timeout=self.debounce_time, record_history_chains=False)
        async def collect_messages(
            controller: SessionController,
            ev: AstrMessageEvent,
        ):
            nonlocal buffer, image_urls
            
            text = (ev.message_str or "").strip()
            
            # 检查是否包含图片
            has_image = False
            for component in ev.message_obj.message:
                if hasattr(component, '__class__') and component.__class__.__name__ == 'Image':
                    has_image = True
                    if hasattr(component, 'url'):
                        image_urls.append(component.url)
                    elif hasattr(component, 'file'):
                        image_urls.append(component.file)
            
            # 如果既没有文本也没有图片，跳过
            if not text and not has_image:
                return
            
            # 如果有文本，检查是否是指令
            if text and self.is_command(text):
                controller.stop()
                return
            
            # 普通消息或图片：接管处理，阻止后续默认流程
            ev.stop_event()
            
            # 如果有文本，加入缓冲区
            if text:
                buffer.append(text)
            
            # 如果只有图片没有文本，添加占位符
            if has_image and not text:
                buffer.append("[图片]")
            
            # 重置超时时间
            controller.keep(timeout=self.debounce_time, reset_timeout=True)
        
        try:
            # 启动会话控制器，第一条消息也会进入 collect_messages
            await collect_messages(event)
            # 如果正常返回（没有超时），说明被 controller.stop() 停止了
            return
            
        except TimeoutError:
            # 超时：合并并发送给 LLM
            merged_message = self.merge_separator.join(buffer).strip()
            if not merged_message:
                return

            logger.info(f"[消息防抖动] 防抖超时，合并了 {len(buffer)} 条消息，图片数: {len(image_urls)}")
            
            # 接管这轮消息
            event.stop_event()
            
            # 获取 LLM 提供商
            umo = event.unified_msg_origin
            provider = self.context.get_using_provider(umo=umo)
            if not provider:
                logger.warning(f"[消息防抖动] 未找到 LLM 提供商")
                yield event.plain_result("错误：未配置LLM提供商")
                return
            
            # 获取人格设定
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(umo=umo)
                
                if persona:
                    if isinstance(persona, dict):
                        system_prompt = persona.get('prompt') or persona.get('system_prompt')
                    else:
                        system_prompt = getattr(persona, 'prompt', None) or getattr(persona, 'system_prompt', None)
                else:
                    system_prompt = None
                    
            except Exception as e:
                logger.error(f"[消息防抖动] 获取会话人格失败: {e}")
                system_prompt = None
            
            # 获取对话历史
            try:
                conv_mgr = self.context.conversation_manager
                curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                conversation = await conv_mgr.get_conversation(
                    umo,
                    curr_cid,
                    create_if_not_exists=True
                )
                
                if conversation and conversation.history:
                    context_history = json.loads(conversation.history)
                else:
                    context_history = []
            except Exception as e:
                logger.warning(f"[消息防抖动] 获取对话历史失败: {e}")
                context_history = []
            
            # 调用 LLM
            try:
                response = await provider.text_chat(
                    prompt=merged_message,
                    context=context_history,
                    system_prompt=system_prompt,
                    image_urls=image_urls if image_urls else None
                )
                
                # 获取响应文本
                response_text = (
                    getattr(response, 'completion_text', None)
                    or getattr(response, 'result', None)
                    or getattr(response, 'content', None)
                    or getattr(response, 'text', None)
                    or getattr(response, 'message', None)
                    or str(response)
                )
                
                if not response_text:
                    logger.error(f"[消息防抖动] LLM 响应为空")
                    yield event.plain_result("抱歉，AI 没有返回有效响应。")
                    return
                
                # 更新对话历史
                try:
                    context_history.append({"role": "user", "content": merged_message})
                    context_history.append({"role": "assistant", "content": response_text})
                    await conv_mgr.update_conversation(
                        umo,
                        curr_cid,
                        history=context_history
                    )
                except Exception as e:
                    logger.warning(f"[消息防抖动] 更新对话历史失败: {e}")
                
                # 发送回复
                yield event.plain_result(response_text)
                
            except Exception as e:
                logger.error(f"[消息防抖动] LLM 请求失败: {e}", exc_info=True)
                yield event.plain_result(f"处理消息时出错: {str(e)}")
        
        except Exception as e:
            logger.error(f"[消息防抖动] 插件内部错误: {e}", exc_info=True)
            event.stop_event()
            yield event.plain_result(f"插件内部错误: {str(e)}")
