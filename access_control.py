"""ID 黑/白名单访问控制模块。"""

from typing import Any, Iterable, Set

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

MODE_DEBOUNCE = "debounce"
MODE_IMMEDIATE = "immediate"
MODE_DENY = "deny"


class IDAccessControl:
    """基于发送者 ID、会话 ID 或 unified_msg_origin 的黑/白名单过滤。

    配置键名沿用 livingmemory 的白名单命名（enable_id_white_list / id_whitelist），
    黑名单通过 blacklist_mode 选择处理方式（disable / immediate / skip）。
    黑名单优先于白名单；名单为空时视为未启用，不过滤任何用户。

    三种处理模式：
    - debounce：正常参与防抖合并。
    - immediate：不合并等待，收到后立即处理（黑名单命中）。
    - deny：完全不处理，直接放行（白名单未命中或黑名单 skip）。
    """

    def __init__(self, config: dict):
        self.enable_id_white_list = bool(config.get("enable_id_white_list", False))
        self.id_whitelist = self._normalize_id_list(config.get("id_whitelist", []))
        self.id_blacklist = self._normalize_id_list(config.get("id_blacklist", []))
        raw_mode = config.get("blacklist_mode")
        if raw_mode in ("disable", "immediate", "skip"):
            self.blacklist_mode = raw_mode
        elif raw_mode == "none":
            # 兼容早期版本的黑名单模式命名
            self.blacklist_mode = "disable"
        elif config.get("enable_id_black_list", False):
            # 兼容早期 fork 版本配置：显式开启 enable_id_black_list 时等价于 immediate
            self.blacklist_mode = "immediate"
        else:
            self.blacklist_mode = "disable"

    @staticmethod
    def _normalize_id_list(items: Iterable[Any]) -> Set[str]:
        """将配置中的 ID 列表归一化为去除空白后的字符串集合，兼容数字/字符串混用。"""
        result = set()
        for item in items or []:
            text = str(item).strip()
            if text:
                result.add(text)
        return result

    def describe(self) -> str:
        """返回启动日志用的访问控制状态描述。"""
        parts = []
        if self.enable_id_white_list and self.id_whitelist:
            parts.append(f"白名单×{len(self.id_whitelist)}")
        if self.blacklist_mode != "disable" and self.id_blacklist:
            parts.append(f"黑名单×{len(self.id_blacklist)}({self.blacklist_mode})")
        return "+".join(parts) if parts else "关闭"

    def get_mode(self, event: AstrMessageEvent) -> str:
        """判断该会话的处理模式。

        匹配字段为发送者 ID、会话 ID 与 unified_msg_origin，任一命中即算名单命中。
        """
        if self.blacklist_mode == "disable" and not (self.enable_id_white_list and self.id_whitelist):
            return MODE_DEBOUNCE

        try:
            candidates = {
                str(event.get_sender_id() or "").strip(),
                str(event.get_session_id() or "").strip(),
                event.unified_msg_origin,
            }
        except Exception as exc:
            # 名单判断失败时放行，避免平台事件差异导致消息处理中断
            logger.warning(f"[消息防抖动] 读取事件 ID 失败，按不限制处理: {exc}")
            return MODE_DEBOUNCE
        candidates.discard("")

        if self.blacklist_mode != "disable" and self.id_blacklist and candidates & self.id_blacklist:
            if self.blacklist_mode == "skip":
                logger.info(f"[消息防抖动] ID 命中黑名单，跳过插件处理 - 用户: {event.unified_msg_origin}")
                return MODE_DENY
            logger.info(f"[消息防抖动] ID 命中黑名单，立即处理（不合并） - 用户: {event.unified_msg_origin}")
            return MODE_IMMEDIATE

        if self.enable_id_white_list and self.id_whitelist and not (candidates & self.id_whitelist):
            logger.info(f"[消息防抖动] ID 不在白名单中，跳过防抖合并 - 用户: {event.unified_msg_origin}")
            return MODE_DENY

        return MODE_DEBOUNCE
