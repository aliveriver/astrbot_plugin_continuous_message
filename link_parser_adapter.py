from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from astrbot.api import logger


class LinkParserAdapter:
    def __init__(self, plugin_config: dict[str, Any]):
        self.plugin_config = plugin_config
        self.enabled = bool(plugin_config.get("enable_link_parsing", True))
        self.disabled_reason = ""
        self.merge_images = bool(plugin_config.get("link_parser_merge_images", True))
        self.max_links = max(0, int(plugin_config.get("link_parser_max_links", 3)))
        self.max_text_length = max(50, int(plugin_config.get("link_parser_max_text_length", 600)))
        self.section_prefix = str(plugin_config.get("link_parser_prefix", "[链接解析]")).strip() or "[链接解析]"
        self.service = None
        self._init_service()

    def _init_service(self) -> None:
        if not self.enabled:
            return
        try:
            module = importlib.import_module(".lite_link_parser", package=__package__)
            self.service = module.LiteLinkParserService(
                {
                    "timeout": float(self.plugin_config.get("link_parser_timeout", 12.0)),
                    "proxy": (self.plugin_config.get("link_parser_proxy", "") or "").strip() or None,
                    "cookie_dir": str(Path(__file__).resolve().parent / "data" / "cookies"),
                    "sites": {
                        "bilibili": {"enable": True, "cookies": self.plugin_config.get("link_parser_bilibili_cookies", "") or ""},
                        "douyin": {"enable": True, "cookies": self.plugin_config.get("link_parser_douyin_cookies", "") or ""},
                        "ncm": {"enable": True, "cookies": self.plugin_config.get("link_parser_ncm_cookies", "") or ""},
                        "weibo": {"enable": True, "cookies": self.plugin_config.get("link_parser_weibo_cookies", "") or ""},
                        "xiaoheihe": {"enable": True, "cookies": self.plugin_config.get("link_parser_xiaoheihe_cookies", "") or ""},
                        "xhs": {"enable": True, "cookies": self.plugin_config.get("link_parser_xhs_cookies", "") or ""},
                    },
                }
            )
            logger.info("[link_parser] initialized successfully")
        except Exception as exc:
            self.enabled = False
            self.disabled_reason = str(exc)
            logger.warning(f"[link_parser] disabled, dependency or init failed: {exc}")

    async def close(self) -> None:
        if self.service:
            await self.service.close()

    async def enrich(self, text: str, image_urls: list[str]) -> tuple[str, list[str]]:
        if not text or self.max_links <= 0:
            return text, image_urls
        if not self.enabled or not self.service:
            if self.disabled_reason:
                logger.debug(f"[link_parser] skipped because disabled: {self.disabled_reason}")
            return text, image_urls

        matches = self.service.find_matches(text, self.max_links)
        if not matches:
            logger.debug("[link_parser] no supported links matched in merged text")
            return text, image_urls
        logger.info(f"[link_parser] matched {len(matches)} link(s) for enrichment")

        sections: list[str] = []
        merged_images = list(image_urls)
        seen_images = {url for url in merged_images if url}

        for item in matches:
            try:
                result = await item.parser.parse(item.keyword, item.match)
            except Exception as exc:
                logger.warning(f"[link_parser] parse failed for {item.raw}: {exc}")
                continue

            section = self._format_result(result)
            if section:
                sections.append(section)

            if self.merge_images:
                for content in result.image_contents:
                    if content.url and content.url not in seen_images:
                        seen_images.add(content.url)
                        merged_images.append(content.url)
                if not result.image_contents:
                    for content in result.video_contents:
                        if content.cover_url and content.cover_url not in seen_images:
                            seen_images.add(content.cover_url)
                            merged_images.append(content.cover_url)

        if not sections:
            logger.debug("[link_parser] no parse result could be appended")
            return text, merged_images

        enriched = f"{text.strip()}\n\n{self.section_prefix}\n" + "\n\n".join(sections)
        logger.info(f"[link_parser] enrichment appended {len(sections)} parsed section(s)")
        return enriched.strip(), merged_images

    def _format_result(self, result) -> str:
        lines: list[str] = [f"平台: {result.platform.display_name}"]
        if result.author and result.author.name:
            lines.append(f"作者: {result.author.name}")
        if result.title:
            lines.append(f"标题: {self._truncate(result.title, 160)}")
        if result.text:
            lines.append(f"正文: {self._truncate(result.text, self.max_text_length)}")
        if result.extra_info:
            lines.append(f"补充: {self._truncate(result.extra_info, 240)}")
        if result.image_contents:
            lines.append(f"图片: {len(result.image_contents)} 张")
        if result.video_contents:
            lines.append(f"视频: {len(result.video_contents)} 个")
        if result.audio_contents:
            lines.append(f"音频: {len(result.audio_contents)} 个")
        if result.url:
            lines.append(f"链接: {result.url}")
        if result.repost:
            repost_text = result.repost.title or result.repost.text or result.repost.url or "转发内容"
            lines.append(f"转发: {self._truncate(repost_text, 120)}")
        return "\n".join(lines).strip()

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        value = " ".join(text.split())
        if len(value) <= limit:
            return value
        return value[: max(limit - 3, 1)].rstrip() + "..."
