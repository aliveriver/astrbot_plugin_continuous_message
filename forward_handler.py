"""
合并转发与引用消息处理模块

负责检测、提取合并转发消息和引用消息的内容。
"""
import json
from typing import List, Tuple, Optional

from astrbot.api import logger
import astrbot.api.message_components as Comp

# 检查是否为 aiocqhttp 平台
try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


class ForwardHandler:
    """合并转发与引用消息处理器"""
    
    def __init__(self, reply_format: str):
        self.reply_format = reply_format

    async def detect_forward_message(self, event: 'AiocqhttpMessageEvent') -> Optional[str]:
        """
        检测消息中是否包含合并转发消息
        
        支持两种场景：
        1. 用户直接发送合并转发消息
        2. 用户回复了一条合并转发消息
        
        Returns:
            合并转发消息的ID，如果没有则返回None
        """
        # 场景1: 直接发送的合并转发
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Forward):
                return seg.id
        
        # 场景2: 回复的合并转发
        reply_seg = None
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Reply):
                reply_seg = seg
                break
        
        if reply_seg:
            try:
                client = event.bot
                original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                
                if original_msg and 'message' in original_msg:
                    original_message_chain = original_msg['message']
                    if isinstance(original_message_chain, list):
                        for segment in original_message_chain:
                            if isinstance(segment, dict) and segment.get("type") == "forward":
                                return segment.get("data", {}).get("id")
            except Exception as e:
                logger.debug(f"[消息防抖动] 获取被回复消息失败: {e}")
        
        return None

    async def extract_reply_content(
        self,
        event: 'AiocqhttpMessageEvent'
    ) -> Tuple[str, List[str]]:
        """
        提取被引用的普通消息内容（非合并转发）
        
        Returns:
            (格式化的文本内容, 图片URL列表)
        """
        # 查找Reply组件
        reply_seg = None
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Reply):
                reply_seg = seg
                break
        
        if not reply_seg:
            return "", []
        
        try:
            client = event.bot
            original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
            
            if not original_msg or 'message' not in original_msg:
                return "", []
            
            # 获取发送者信息
            sender_id = original_msg.get('sender', {}).get('user_id')
            sender_name = original_msg.get('sender', {}).get('nickname', '未知用户')
            
            # 检查发送者是否是bot自己
            bot_id = None
            try:
                if hasattr(event, 'message_obj') and hasattr(event.message_obj, 'self_id'):
                    bot_id = event.message_obj.self_id
                    if bot_id:
                        try:
                            bot_id = int(bot_id) if isinstance(bot_id, str) else bot_id
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                logger.debug(f"[消息防抖动] 获取bot ID失败: {e}")
                bot_id = None
            
            # 确保sender_id也是整数类型
            if isinstance(sender_id, str):
                try:
                    sender_id = int(sender_id)
                except (ValueError, TypeError):
                    pass
            
            is_bot_message = (bot_id is not None and sender_id == bot_id)
            
            logger.debug(f"[消息防抖动] 引用消息判断 | sender_id: {sender_id}, bot_id: {bot_id}, is_bot_message: {is_bot_message}, sender_name: {sender_name}")
            
            # 解析消息内容
            original_message_chain = original_msg['message']
            content_chain = self._parse_raw_content(original_message_chain)
            
            # 提取文本和图片
            text_parts = []
            image_urls = []
            
            for segment in content_chain:
                if not isinstance(segment, dict):
                    continue
                
                seg_type = segment.get("type")
                seg_data = segment.get("data", {})
                
                if seg_type == "text":
                    text = seg_data.get("text", "")
                    if text:
                        text_parts.append(text)
                
                elif seg_type == "image":
                    url = seg_data.get("url")
                    if url:
                        image_urls.append(url)
                        text_parts.append("[图片]")
            
            full_text = "".join(text_parts).strip()
            
            if not full_text:
                return "", image_urls
            
            # 格式化文本，sender 属性已足以标明发送者身份（包括bot自己），无需额外提示
            formatted_text = self.reply_format.format(sender_name=sender_name, full_text=full_text)
            
            return formatted_text, image_urls
            
        except Exception as e:
            logger.debug(f"[消息防抖动] 提取引用消息失败: {e}")
            return "", []

    async def extract_forward_content(
        self, 
        event: 'AiocqhttpMessageEvent', 
        forward_id: str
    ) -> Tuple[str, List[str]]:
        """
        从合并转发消息中提取文本和图片URL
        
        Returns:
            (格式化的文本内容, 图片URL列表)
        """
        client = event.bot
        
        try:
            forward_data = await client.api.call_action('get_forward_msg', id=forward_id)
        except Exception as e:
            logger.error(f"[消息防抖动] 调用 get_forward_msg API 失败: {e}")
            raise ValueError("获取合并转发内容失败，可能是消息已过期或API问题")

        if not forward_data or "messages" not in forward_data:
            logger.error(f"[消息防抖动] forward_data 无效或缺少 messages 字段")
            raise ValueError("获取到的合并转发内容为空")
        
        if len(forward_data['messages']) == 0:
            logger.warning(f"[消息防抖动] NapCat返回的messages为空数组，可能是API限制或配置问题")
            return "", []

        extracted_texts = []
        image_urls = []

        for message_node in forward_data["messages"]:
            logger.debug(f"[消息防抖动] 处理消息节点: {message_node}")
            sender_name = message_node.get("sender", {}).get("nickname", "未知用户")
            
            raw_content = message_node.get("message") or message_node.get("content", [])
            content_chain = self._parse_raw_content(raw_content)
            
            node_text_parts = []
            for segment in content_chain:
                if not isinstance(segment, dict):
                    continue
                    
                seg_type = segment.get("type")
                seg_data = segment.get("data", {})
                
                if seg_type == "text":
                    text = seg_data.get("text", "")
                    if text:
                        node_text_parts.append(text)
                
                elif seg_type == "image":
                    url = seg_data.get("url")
                    if url:
                        image_urls.append(url)
                        node_text_parts.append("[图片]")
            
            full_node_text = "".join(node_text_parts).strip()
            logger.debug(f"[消息防抖动] 节点提取的文本: '{full_node_text}'")
            if full_node_text:
                extracted_texts.append(f"{sender_name}: {full_node_text}")
            else:
                logger.warning(f"[消息防抖动] 节点没有文本内容，跳过")

        inner = "\n".join(extracted_texts)
        logger.debug(f"[消息防抖动] 合并转发提取完成 | 节点数: {len(forward_data['messages'])} | 有效文本节点数: {len(extracted_texts)} | 图片数: {len(image_urls)} | 内容预览: {inner[:200]!r}")
        wrapped = f"<forward_content>\n{inner}\n</forward_content>" if inner else ""
        return wrapped, image_urls

    def _parse_raw_content(self, raw_content) -> List[dict]:
        """
        解析原始消息内容
        
        支持的格式：
        1. 列表形式: [{"type": "text", "data": {...}}, ...]
        2. JSON字符串: '[{"type": "text", ...}]'
        3. 纯文本字符串: "hello world"
        
        Returns:
            标准化的消息链列表
        """
        if isinstance(raw_content, list):
            return raw_content
        
        if isinstance(raw_content, str):
            try:
                parsed_content = json.loads(raw_content)
                if isinstance(parsed_content, list):
                    return parsed_content
            except (json.JSONDecodeError, TypeError):
                pass
            
            return [{"type": "text", "data": {"text": raw_content}}]
        
        return []
