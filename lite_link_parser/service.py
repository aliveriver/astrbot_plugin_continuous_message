from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from .base import BaseLiteParser
from .parsers import (
    BilibiliLiteParser,
    NCMLiteParser,
    XiaoheiheLiteParser,
    XHSLiteParser,
    LofterLiteParser,
    QQMusicLiteParser,
    ZhihuLiteParser,
)


@dataclass(slots=True)
class MatchedLink:
    parser: BaseLiteParser
    keyword: str
    match: Any
    start: int
    end: int
    raw: str


class LiteLinkParserService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.parsers: list[BaseLiteParser] = []
        self.key_pattern_list: list[tuple[BaseLiteParser, str, Any]] = []
        self._build()

    def _build(self) -> None:
        supported = [
            BilibiliLiteParser,
            NCMLiteParser,
            XiaoheiheLiteParser,
            XHSLiteParser,
            LofterLiteParser,
            QQMusicLiteParser,
            ZhihuLiteParser,
        ]
        for parser_cls in supported:
            site_config = self.config["sites"].get(parser_cls.platform.name, {})
            if site_config and not site_config.get("enable", True):
                continue
            parser = parser_cls(self.config)
            self.parsers.append(parser)
            for keyword, pattern in parser_cls._key_patterns:
                self.key_pattern_list.append((parser, keyword, pattern))

        self.key_pattern_list.sort(key=lambda item: -len(item[1]))
        logger.info(
            "[link_parser] enabled platforms: "
            + (", ".join(parser.platform.name for parser in self.parsers) or "none")
        )

    async def close(self) -> None:
        for parser in self.parsers:
            await parser.close_session()

    def find_matches(self, text: str, max_links: int) -> list[MatchedLink]:
        results: list[MatchedLink] = []
        for parser, keyword, pattern in self.key_pattern_list:
            if keyword not in text:
                continue
            for match in pattern.finditer(text):
                results.append(
                    MatchedLink(
                        parser=parser,
                        keyword=keyword,
                        match=match,
                        start=match.start(),
                        end=match.end(),
                        raw=match.group(0).strip(),
                    )
                )

        results.sort(key=lambda item: (item.start, -(item.end - item.start)))
        filtered: list[MatchedLink] = []
        seen_raw: set[str] = set()
        last_end = -1
        for item in results:
            if item.start < last_end:
                continue
            if item.raw in seen_raw:
                continue
            filtered.append(item)
            seen_raw.add(item.raw)
            last_end = item.end
            if len(filtered) >= max_links:
                break
        return filtered
