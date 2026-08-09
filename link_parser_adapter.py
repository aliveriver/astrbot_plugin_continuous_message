from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from astrbot.api import logger


class LinkParserAdapter:
    _LINK_PLATFORM_LABELS = {
        "bilibili": "B站",
        "ncm": "网易云音乐",
        "xhs": "小红书",
        "xiaoheihe": "小黑盒",
        "lofter": "LOFTER",
        "qqmusic": "QQ音乐",
        "zhihu": "知乎",
    }

    def __init__(self, plugin_config: dict[str, Any]):
        self.plugin_config = plugin_config
        self.enabled = bool(plugin_config.get("enable_link_parsing", True))
        self.disabled_reason = ""
        self.merge_images = bool(plugin_config.get("link_parser_merge_images", True))
        self.max_links = max(0, int(plugin_config.get("link_parser_max_links", 3)))
        self.max_text_length = max(50, int(plugin_config.get("link_parser_max_text_length", 600)))
        self.disabled_platforms = self._normalize_platform_set(
            plugin_config.get("link_parser_disabled_platforms", [])
        )
        self.success_prompt = str(
            plugin_config.get(
                "link_parser_success_prompt",
                plugin_config.get("link_parser_prefix", "[链接解析]"),
            )
        ).strip()
        self.failure_prompt = str(
            plugin_config.get("link_parser_failure_prompt", "[链接解析失败]")
        ).strip()
        self.service = None
        self._init_service()

    @staticmethod
    def _normalize_platform_set(value: Any) -> set[str]:
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value]
        else:
            items = []
        return {item.lower() for item in items if item}

    @staticmethod
    def _compose_section(prompt: str, body: str) -> str:
        clean_body = body.strip()
        clean_prompt = (prompt or "").strip()
        if clean_prompt and clean_body:
            return f"{clean_prompt}\n{clean_body}"
        return clean_prompt or clean_body

    def _init_service(self) -> None:
        if not self.enabled:
            return
        try:
            if self.disabled_platforms:
                disabled_text = ", ".join(
                    self._LINK_PLATFORM_LABELS.get(name, name) for name in sorted(self.disabled_platforms)
                )
                logger.info(f"[link_parser] disabled platforms: {disabled_text}")
            module = importlib.import_module(".lite_link_parser", package=__package__)
            self.service = module.LiteLinkParserService(
                {
                    "timeout": float(self.plugin_config.get("link_parser_timeout", 12.0)),
                    "proxy": (self.plugin_config.get("link_parser_proxy", "") or "").strip() or None,
                    "cookie_dir": str(Path(__file__).resolve().parent / "data" / "cookies"),
                    "sites": {
                        "bilibili": {"enable": "bilibili" not in self.disabled_platforms},
                        "ncm": {"enable": "ncm" not in self.disabled_platforms},
                        "xiaoheihe": {"enable": "xiaoheihe" not in self.disabled_platforms},
                        "xhs": {"enable": "xhs" not in self.disabled_platforms},
                        "lofter": {"enable": "lofter" not in self.disabled_platforms},
                        "qqmusic": {"enable": "qqmusic" not in self.disabled_platforms},
                        "zhihu": {"enable": "zhihu" not in self.disabled_platforms},
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

        success_sections: list[str] = []
        failure_sections: list[str] = []
        merged_images = list(image_urls)
        seen_images = {url for url in merged_images if url}

        for item in matches:
            try:
                result = await item.parser.parse(item.keyword, item.match)
            except Exception as exc:
                logger.warning(f"[link_parser] parse failed for {item.raw}: {exc}")
                failure_sections.append(
                    self._format_failure(item.parser.platform.display_name, item.raw, str(exc))
                )
                continue

            section = self._format_result(result)
            if section:
                success_sections.append(section)

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

        appended_sections: list[str] = []
        if success_sections:
            appended_sections.append(
                self._compose_section(self.success_prompt, "\n\n".join(success_sections))
            )
        if failure_sections:
            appended_sections.append(
                self._compose_section(self.failure_prompt, "\n\n".join(failure_sections))
            )

        if not appended_sections:
            logger.debug("[link_parser] no parse result could be appended")
            return text, merged_images

        enriched = f"{text.strip()}\n\n" + "\n\n".join(section for section in appended_sections if section)
        logger.info(
            "[link_parser] enrichment appended "
            f"{len(success_sections)} success section(s) and {len(failure_sections)} failure section(s)"
        )
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

    def _format_failure(self, platform_name: str, raw_url: str, reason: str) -> str:
        lines = [f"平台: {platform_name}", f"链接: {raw_url}"]
        clean_reason = self._truncate(" ".join((reason or "").split()), 120)
        if clean_reason:
            lines.append(f"原因: {clean_reason}")
        return "\n".join(lines).strip()

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        value = " ".join(text.split())
        if len(value) <= limit:
            return value
        return value[: max(limit - 3, 1)].rstrip() + "..."
