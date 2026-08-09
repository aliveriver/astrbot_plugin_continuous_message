"""
消息解析模块

负责消息内容的解析、图片提取、事件重构和输入状态检测。
"""
import json
import os
import re
from urllib.parse import parse_qs, urlparse
from typing import List, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

# 检查是否为 aiocqhttp 平台
try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


class MessageParser:
    """消息解析器：提供消息文本/图片提取、事件重构、输入状态检测等功能"""

    _URL_KEY_HINTS = {"jumpurl", "qqdocurl", "url", "musicurl"}
    _SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")
    _HOST_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$")
    _CARD_PLATFORM_LABELS = {
        "bilibili": "B站",
        "nga": "NGA",
        "ncm": "网易云音乐",
        "tieba": "百度贴吧",
        "xhs": "小红书",
        "xiaoheihe": "小黑盒",
        "zhihu": "知乎",
    }
    
    def __init__(self, image_component=None, plain_component=None, plugin_config=None):
        self._ImageComponent = image_component
        self._PlainComponent = plain_component
        config = plugin_config or {}
        self.enable_qq_card_parsing = bool(config.get("enable_qq_card_parsing", True))
        self.qq_card_prompt = str(config.get("qq_card_prompt", "[卡片链接]")).strip()
        self.qq_card_disabled_platforms = self._normalize_platform_set(
            config.get("qq_card_disabled_platforms", [])
        )

    @staticmethod
    def _normalize_platform_set(value) -> set[str]:
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value]
        else:
            items = []
        return {item.lower() for item in items if item}

    @staticmethod
    def _append_prompt_line(prompt: str, url: str) -> str:
        clean_prompt = (prompt or "").strip()
        return f"{clean_prompt} {url}".strip() if clean_prompt else url

    def _safe_json_loads(self, value, source: str = "qq_card"):
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except Exception as exc:
            stripped = value.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                snippet = " ".join(stripped.split())[:160]
                logger.warning(
                    f"[消息防抖动] QQ 卡片内容反序列化失败 | source={source} | error={exc} | payload={snippet}"
                )
            return value

    def _normalize_url(self, value: str) -> str:
        if not value:
            return ""
        url = value.strip()
        if not url:
            return ""
        if url.startswith("//"):
            return f"https:{url}"
        if self._SCHEME_PATTERN.match(url):
            return url
        if self._HOST_PATH_PATTERN.match(url):
            return f"https://{url}"
        return ""

    def _rule_xiaoheihe_bbs_share(self, host: str, path: str, query: dict) -> str:
        if host == "api.xiaoheihe.cn" and path == "/v3/bbs/app/api/web/share":
            link_id = (query.get("link_id") or [""])[0].strip()
            if link_id:
                return f"https://www.xiaoheihe.cn/app/bbs/link/{link_id}"
        return ""

    def _rule_xiaoheihe_game_share(self, host: str, path: str, query: dict) -> str:
        if host == "api.xiaoheihe.cn" and path == "/game/share_game_detail":
            appid = (query.get("appid") or [""])[0].strip()
            game_type = (query.get("game_type") or ["pc"])[0].strip().lower() or "pc"
            if appid:
                return f"https://www.xiaoheihe.cn/app/topic/game/{game_type}/{appid}"
        return ""

    def _rule_tieba_post_share(self, host: str, path: str, query: dict) -> str:
        if host in {"tieba.baidu.com", "www.tieba.baidu.com"}:
            matched = re.match(r"^/p/(\d+)", path)
            if matched:
                return f"https://tieba.baidu.com/p/{matched.group(1)}"
        return ""

    def _rule_bilibili_share(self, host: str, path: str, query: dict) -> str:
        if host in {"www.bilibili.com", "bilibili.com", "m.bilibili.com"}:
            matched = re.match(r"^/(?:video/)?(?P<video_id>BV[0-9A-Za-z]{10}|av\d+)", path)
            if matched:
                page_num = (query.get("p") or [""])[0].strip()
                canonical = f"https://www.bilibili.com/video/{matched.group('video_id')}"
                if page_num.isdigit() and int(page_num) > 1:
                    canonical += f"?p={page_num}"
                return canonical
        return ""

    def _rule_xhs_share(self, host: str, path: str, query: dict, original_url: str) -> str:
        if host not in {"www.xiaohongshu.com", "xiaohongshu.com"}:
            return ""
        matched = re.match(r"^/(?:discovery/item|explore)/(?P<note_id>[0-9A-Za-z]+)", path)
        if not matched:
            return ""
        note_id = matched.group("note_id")
        if not note_id:
            return ""

        query_items = []
        for key, values in query.items():
            for value in values:
                query_items.append(f"{key}={value}")
        query_text = "&".join(query_items)
        if query_text:
            return f"https://www.xiaohongshu.com/discovery/item/{note_id}?{query_text}"
        return original_url

    def _rule_nga_share(self, host: str, path: str, query: dict) -> str:
        if host not in {"ngabbs.com", "nga.178.com", "bbs.nga.cn"}:
            return ""
        if path != "/read.php":
            return ""
        tid = (query.get("tid") or [""])[0].strip()
        if tid.isdigit():
            return f"https://ngabbs.com/read.php?tid={tid}"
        return ""

    def _rule_ncm_song_share(self, host: str, path: str, query: dict) -> str:
        if host == "y.music.163.com" and path == "/m/song":
            song_id = (query.get("id") or [""])[0].strip()
            if song_id.isdigit():
                return f"https://music.163.com/#/song?id={song_id}"
        return ""

    def _rule_zhihu_share(self, host: str, path: str, query: dict) -> str:
        if host == "zhuanlan.zhihu.com":
            article_matched = re.match(r"^/p/(\d+)", path)
            if article_matched:
                return f"https://zhuanlan.zhihu.com/p/{article_matched.group(1)}"

        if host in {"www.zhihu.com", "zhihu.com"}:
            answer_matched = re.match(r"^/question/(\d+)/answer/(\d+)", path)
            if answer_matched:
                qid, aid = answer_matched.groups()
                return f"https://www.zhihu.com/question/{qid}/answer/{aid}"

            question_matched = re.match(r"^/question/(\d+)", path)
            if question_matched:
                return f"https://www.zhihu.com/question/{question_matched.group(1)}"
        return ""

    def _is_wrapper_share_url(self, url: str) -> bool:
        """识别QQ卡片常见中转壳链接，避免优先输出不可解析链接。"""
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            path = parsed.path or ""
            return host in {"m.q.qq.com", "q.qq.com"} and path.startswith("/a/s/")
        except Exception:
            return False

    def _identify_card_platform(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            path = parsed.path or ""
        except Exception:
            return ""

        if host in {"www.bilibili.com", "bilibili.com", "m.bilibili.com", "b23.tv", "bili2233.cn"}:
            return "bilibili"
        if host in {"www.xiaohongshu.com", "xiaohongshu.com", "xhslink.com"}:
            return "xhs"
        if host in {"api.xiaoheihe.cn", "www.xiaoheihe.cn", "xiaoheihe.cn"}:
            return "xiaoheihe"
        if host in {"tieba.baidu.com", "www.tieba.baidu.com"} and re.match(r"^/p/\d+", path):
            return "tieba"
        if host in {"ngabbs.com", "nga.178.com", "bbs.nga.cn"}:
            return "nga"
        if host in {"y.music.163.com", "music.163.com", "163cn.tv"}:
            return "ncm"
        if host in {"www.zhihu.com", "zhihu.com", "zhuanlan.zhihu.com"}:
            return "zhihu"
        if host in {"www.lofter.com", "lofter.com", "lofter.net"}:
            return "lofter"
        if host in {"y.qq.com", "i.y.qq.com", "c.y.qq.com"}:
            return "qqmusic"
        return ""

    def _needs_card_canonicalization(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            path = parsed.path or ""
        except Exception:
            return False

        if host in {"m.q.qq.com", "q.qq.com"} and path.startswith("/a/s/"):
            return True
        if host == "api.xiaoheihe.cn" and path in {"/v3/bbs/app/api/web/share", "/game/share_game_detail"}:
            return True
        if host in {"b23.tv", "bili2233.cn", "163cn.tv", "xhslink.com"}:
            return True
        if host == "y.music.163.com" and path == "/m/song":
            return True
        return False

    def _process_card_url(self, normalized_url: str) -> str:
        canonical = self._canonicalize_known_share_url(normalized_url)
        platform = self._identify_card_platform(canonical) or self._identify_card_platform(normalized_url)

        if not platform:
            logger.debug(f"[消息防抖动] 提取到 URL 但无法识别平台: {normalized_url}")
        elif canonical == normalized_url and self._needs_card_canonicalization(normalized_url):
            logger.debug(f"[消息防抖动] QQ 卡片链接规范化失败，保留原始链接: {normalized_url}")

        return self._filter_card_url(normalized_url, canonical)

    def _filter_card_url(self, original_url: str, canonical_url: str) -> str:
        platform = self._identify_card_platform(canonical_url) or self._identify_card_platform(original_url)
        if platform and platform in self.qq_card_disabled_platforms:
            display_name = self._CARD_PLATFORM_LABELS.get(platform, platform)
            logger.debug(
                f"[消息防抖动] 跳过已屏蔽的 QQ 卡片平台: {display_name} ({canonical_url or original_url})"
            )
            return ""
        return canonical_url or original_url

    def _apply_share_url_rules(self, host: str, path: str, query: dict, fallback_url: str) -> str:
        rules = (
            self._rule_xiaoheihe_bbs_share,
            self._rule_xiaoheihe_game_share,
            self._rule_tieba_post_share,
            self._rule_bilibili_share,
            self._rule_xhs_share,
            self._rule_nga_share,
            self._rule_ncm_song_share,
            self._rule_zhihu_share,
        )
        for rule in rules:
            if getattr(rule, "__name__", "") == "_rule_xhs_share":
                result = rule(host, path, query, fallback_url)
            else:
                result = rule(host, path, query)
            if result:
                return result
        return fallback_url

    def _canonicalize_known_share_url(self, url: str) -> str:
        """统一入口：将卡片分享链接规范化为更稳定、可打开的网页链接。"""
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            path = parsed.path or ""
            query = parse_qs(parsed.query)
            return self._apply_share_url_rules(host, path, query, url)
        except Exception:
            return url

    def _extract_urls_from_json_payload(self, payload) -> List[str]:
        payload = self._safe_json_loads(payload, source="qq_card_root")
        if isinstance(payload, dict) and "data" in payload:
            payload["data"] = self._safe_json_loads(payload.get("data"), source="qq_card_data")

        extracted: List[str] = []

        def walk(node):
            if isinstance(node, dict):
                for key, val in node.items():
                    key_lower = str(key).lower()
                    if key_lower in self._URL_KEY_HINTS and isinstance(val, str):
                        normalized = self._normalize_url(val)
                        if normalized:
                            filtered = self._process_card_url(normalized)
                            if filtered:
                                extracted.append(filtered)
                    walk(val)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)

        # 去重并保持原始顺序
        deduped = []
        seen = set()
        for u in extracted:
            if u not in seen:
                seen.add(u)
                deduped.append(u)

        non_wrappers = [u for u in deduped if not self._is_wrapper_share_url(u)]
        return non_wrappers or deduped

    @staticmethod
    def _dedupe_keep_order(values: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    @staticmethod
    def _raw_get(raw, key, default=None):
        try:
            if isinstance(raw, dict):
                return raw.get(key, default)
            if hasattr(raw, "get"):
                return raw.get(key, default)
        except Exception:
            pass
        return getattr(raw, key, default)

    def _extract_image_urls_from_raw_message(self, message_obj) -> List[str]:
        """Prefer durable image refs from the platform payload over temp files."""
        raw = getattr(message_obj, "raw_message", None)
        raw_chain = self._raw_get(raw, "message", [])
        if not isinstance(raw_chain, list):
            return []

        image_urls = []
        for segment in raw_chain:
            if not isinstance(segment, dict) or segment.get("type") != "image":
                continue
            data = segment.get("data") or {}
            if not isinstance(data, dict):
                continue
            image_ref = data.get("url") or data.get("file")
            if image_ref:
                image_urls.append(str(image_ref))
        return self._dedupe_keep_order(image_urls)

    @staticmethod
    def _build_raw_message_segments(text: str, image_urls: List[str]) -> List[dict]:
        segments = []
        if text:
            segments.append({"type": "text", "data": {"text": text}})
        for image_ref in image_urls or []:
            if image_ref:
                data = {"file": image_ref}
                if str(image_ref).startswith(("http://", "https://")):
                    data["url"] = image_ref
                segments.append({"type": "image", "data": data})
        return segments

    @staticmethod
    def _build_raw_message_text(text: str, image_urls: List[str]) -> str:
        parts = [text] if text else []
        for image_ref in image_urls or []:
            if not image_ref:
                continue
            if str(image_ref).startswith(("http://", "https://")):
                parts.append(f"[CQ:image,file={image_ref},url={image_ref}]")
            else:
                parts.append(f"[CQ:image,file={image_ref}]")
        return "".join(parts)

    def _sync_raw_message(
        self,
        event: AstrMessageEvent,
        text: str,
        image_urls: List[str],
    ):
        message_obj = getattr(event, "message_obj", None)
        raw = getattr(message_obj, "raw_message", None)
        if raw is None:
            return

        raw_segments = self._build_raw_message_segments(text, image_urls)
        raw_text = self._build_raw_message_text(text, image_urls)
        try:
            raw["message"] = raw_segments
            raw["raw_message"] = raw_text
        except Exception:
            try:
                setattr(raw, "message", raw_segments)
                setattr(raw, "raw_message", raw_text)
            except Exception:
                pass

    def is_command(self, message: str, prefixes: list) -> bool:
        """检查消息是否为指令"""
        message = message.strip()
        if not message:
            return False
        for prefix in prefixes:
            if message.startswith(prefix):
                return True
        return False

    def is_typing_event(self, event: AstrMessageEvent) -> bool:
        """检测是否为输入状态通知事件（NapCat input_status）"""
        if not IS_AIOCQHTTP:
            return False
        try:
            raw = getattr(event.message_obj, 'raw_message', None)
            if raw is None:
                return False
            return (
                raw.get('post_type') == 'notice'
                and raw.get('sub_type') == 'input_status'
            )
        except Exception:
            return False

    def get_message_id(self, event: AstrMessageEvent):
        """
        获取消息的 message_id。
        优先从 message_obj.message_id 获取，回退到 raw_message['message_id']。
        返回字符串或整数，调用方可自行做 str() 统一比较。
        """
        try:
            mid = getattr(event.message_obj, 'message_id', None)
            if mid is not None:
                return mid
            raw = getattr(event.message_obj, 'raw_message', None)
            if isinstance(raw, dict):
                return raw.get('message_id')
        except Exception:
            pass
        return None

    def is_recall_event(self, event: AstrMessageEvent) -> bool:
        """
        检测是否为消息撤回通知事件（OneBot v11 friend_recall / group_recall）。
        仅在 aiocqhttp 平台有效。
        """
        if not IS_AIOCQHTTP:
            return False
        try:
            raw = getattr(event.message_obj, 'raw_message', None)
            if not isinstance(raw, dict):
                return False
            return (
                raw.get('post_type') == 'notice'
                and raw.get('notice_type') in ('friend_recall', 'group_recall')
            )
        except Exception:
            return False

    def get_recalled_message_id(self, event: AstrMessageEvent):
        """
        从撤回通知事件中提取被撤回消息的 message_id。
        OneBot v11 撤回事件的 message_id 字段即为被撤回消息的 ID。
        """
        try:
            raw = getattr(event.message_obj, 'raw_message', None)
            if isinstance(raw, dict):
                return raw.get('message_id')
        except Exception:
            pass
        return None

    def parse_message(self, message_obj) -> Tuple[str, bool, List[str]]:
        """
        解析消息对象，提取文本、图片和合并转发信息
        
        Returns:
            (文本内容, 是否包含图片, 图片URL列表)
        """
        text = ""
        has_image = False
        image_urls = []
        card_urls = []
        try:
            if not hasattr(message_obj, "message"):
                return "", False, []
            
            for component in message_obj.message:
                # 跳过Reply组件（引用消息由forward_handler单独处理）
                if component.__class__.__name__ == 'Reply':
                    continue
                
                # 提取文本内容（支持多种属性名）
                if hasattr(component, 'text') and component.text:
                    text += component.text
                elif hasattr(component, 'content') and component.content:
                    text += component.content

                # 提取QQ卡片(json段)中的原始链接
                json_payload = None
                if component.__class__.__name__ == 'Json' and hasattr(component, 'data'):
                    json_payload = component.data
                elif isinstance(component, dict) and component.get('type') == 'json':
                    json_payload = component.get('data')
                if json_payload is not None:
                    try:
                        card_urls.extend(self._extract_urls_from_json_payload(json_payload))
                    except Exception as exc:
                        logger.error(f"[消息防抖动] QQ 卡片解析异常: {exc}")
                
                # 识别图片组件（优先使用 isinstance，后备使用类名检查）
                is_img = False
                if self._ImageComponent and isinstance(component, self._ImageComponent):
                    is_img = True
                elif component.__class__.__name__ == 'Image':
                    is_img = True
                
                # 提取图片URL
                if is_img:
                    has_image = True
                    if hasattr(component, 'url') and component.url:
                        image_urls.append(component.url)
                    elif hasattr(component, 'file') and component.file:
                        image_urls.append(component.file)
        except Exception as exc:
            logger.error(f"[消息防抖动] 消息解析异常: {exc}")

        if self.enable_qq_card_parsing and card_urls:
            deduped_links = self._dedupe_keep_order(card_urls)
            card_text = "\n".join(
                self._append_prompt_line(self.qq_card_prompt, url)
                for url in deduped_links
            )
            text = f"{text}\n{card_text}" if text else card_text

        raw_image_urls = self._extract_image_urls_from_raw_message(message_obj)
        if raw_image_urls:
            image_urls = raw_image_urls
            has_image = True

        return text, has_image, self._dedupe_keep_order(image_urls)

    def _build_image_component(self, image_ref: str, prefer_filesystem: bool = False):
        if not self._ImageComponent or not image_ref:
            return None

        if (
            prefer_filesystem
            and os.path.isfile(image_ref)
            and hasattr(self._ImageComponent, "fromFileSystem")
        ):
            try:
                return self._ImageComponent.fromFileSystem(image_ref)
            except Exception as exc:
                logger.warning(f"[消息防抖动] Image.fromFileSystem 构造失败，回退普通 Image: {exc}")

        if self._SCHEME_PATTERN.match(image_ref) and hasattr(self._ImageComponent, "fromURL"):
            try:
                return self._ImageComponent.fromURL(image_ref)
            except Exception:
                pass

        try:
            return self._ImageComponent(file=image_ref)
        except TypeError:
            return self._ImageComponent(url=image_ref)
        except Exception:
            return None

    def reconstruct_event(
        self,
        event: AstrMessageEvent,
        text: str,
        image_urls: List[str],
        prefer_filesystem_images: bool = False,
    ):
        """
        重构消息事件，将合并后的文本和图片重新组装到事件对象中
        这样事件可以继续传播给后续的插件/框架处理
        """
        event.message_str = text
        if hasattr(event, "message_obj"):
            try:
                event.message_obj.message_str = text
            except Exception:
                pass

        self._sync_raw_message(event, text, image_urls)

        if not self._PlainComponent:
            return

        # 构建消息组件链：文本 + 图片
        chain = []
        if text:
            chain.append(self._PlainComponent(text=text))
        
        # 添加图片组件（兼容不同的 Image 构造函数参数）
        if image_urls and self._ImageComponent:
            for image_ref in image_urls:
                image_component = self._build_image_component(
                    image_ref,
                    prefer_filesystem=prefer_filesystem_images,
                )
                if image_component:
                    chain.append(image_component)
        
        # 更新事件的消息对象
        if hasattr(event.message_obj, "message"):
            try:
                event.message_obj.message = chain
            except Exception:
                pass
