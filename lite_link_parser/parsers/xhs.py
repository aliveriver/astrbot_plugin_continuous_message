from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from msgspec import Struct, convert

from astrbot.api import logger

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform
from ..exception import ParseException
from curl_cffi import requests as curl_requests


class XHSLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="xhs", display_name="小红书")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.headers.update(
            {
                "accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                    "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                )
            }
        )
        self.ios_headers.update(
            {
                "origin": "https://www.xiaohongshu.com",
                "x-requested-with": "XMLHttpRequest",
                "sec-fetch-site": "same-origin",
                "sec-fetch-mode": "cors",
                "sec-fetch-dest": "empty",
            }
        )
        cookie_dir = self.ensure_cookie_dir(self.config["cookie_dir"])
        self.cookiejar = CookieJar(
            cookie_dir,
            name="xhs",
            domain="xiaohongshu.com",
            raw_cookies=self.site_config.get("cookies", ""),
        )
        if self.cookiejar.cookies_str:
            self.headers["cookie"] = self.cookiejar.cookies_str
            self.ios_headers["cookie"] = self.cookiejar.cookies_str

    @handle("xhslink.com", r"xhslink\.com/[A-Za-z0-9._?%&+=/#@-]+")
    async def _parse_short_link(self, searched: re.Match[str]):
        return await self.parse_with_redirect(f"https://{searched.group(0)}", self.ios_headers)

    @handle(
        "xiaohongshu.com",
        r"(explore|discovery/item)/(?P<query>(?P<xhs_id>[0-9a-zA-Z]+)\?[A-Za-z0-9._%&+=/#@-]+)",
    )
    async def _parse_common(self, searched: re.Match[str]):
        xhs_domain = "https://www.xiaohongshu.com"
        query, xhs_id = searched.group("query", "xhs_id")
        try:
            return await self.parse_explore(f"{xhs_domain}/explore/{query}", xhs_id)
        except Exception as exc:
            logger.warning(f"[link_parser:xhs] parse_explore failed, fallback to discovery: {exc}")
            return await self.parse_discovery(f"{xhs_domain}/discovery/item/{query}")

    async def parse_explore(self, url: str, xhs_id: str):
        # 组装代理格式以适配 curl_cffi
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        
        # 使用 curl_cffi 完美伪装成 Chrome 110 浏览器，绕过常规指纹识别
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            response = await session.get(url, headers=self.headers, proxies=proxies)
            html = response.text

        payload = self._extract_initial_state_json(html)
        note_data = payload.get("note", {}).get("noteDetailMap", {}).get(xhs_id, {}).get("note", {})
        if not note_data:
            raise ParseException("xhs note detail missing")

        user = note_data.get("user") or {}
        image_urls = [
            item.get("urlDefault")
            for item in (note_data.get("imageList") or [])
            if isinstance(item, dict) and item.get("urlDefault")
        ]
        video_url = self._extract_video_url(note_data.get("video"))
        contents = []
        if video_url:
            contents.append(self.create_video_content(video_url, image_urls[0] if image_urls else None))
        elif image_urls:
            contents.extend(self.create_image_contents(image_urls))

        return self.result(
            title=note_data.get("title"),
            text=note_data.get("desc"),
            author=self.create_author(user.get("nickname"), user.get("avatar")),
            contents=contents,
            url=url,
        )

    async def parse_discovery(self, url: str):
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        
        # 使用 curl_cffi 伪装并允许重定向
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            response = await session.get(
                url, 
                headers=self.ios_headers, 
                proxies=proxies, 
                allow_redirects=True
            )
            html = response.text

        payload = self._extract_initial_state_json(html)
        note_data = payload.get("noteData")
        if not note_data:
            raise ParseException("xhs noteData missing")
        preload_data = note_data.get("normalNotePreloadData", {})
        
        # --- 核心修复：防止 NoneType 报错的容错机制 ---
        inner_data = note_data.get("data")
        if inner_data is None:
            inner_data = {}
        note_data = inner_data.get("noteData", {})
        # -----------------------------------------------

        if not note_data:
            raise ParseException("xhs noteData.data missing (可能由于视频笔记类型或触发了验证码拦截)")

        user = note_data.get("user") or {}
        image_urls = [
            item.get("url")
            for item in (note_data.get("imageList") or [])
            if isinstance(item, dict) and item.get("url")
        ]
        preload_image_urls = [
            item.get("urlSizeLarge") or item.get("url")
            for item in (preload_data.get("imagesList") or [])
            if isinstance(item, dict) and (item.get("urlSizeLarge") or item.get("url"))
        ]
        video_url = self._extract_video_url(note_data.get("video"))
        contents = []
        if video_url:
            cover_candidates = preload_image_urls or image_urls
            contents.append(self.create_video_content(video_url, cover_candidates[0] if cover_candidates else None))
        elif image_urls:
            contents.extend(self.create_image_contents(image_urls))

        return self.result(
            title=note_data.get("title"),
            author=self.create_author(user.get("nickName"), user.get("avatar")),
            contents=contents,
            text=note_data.get("desc"),
            timestamp=(note_data.get("time") or 0) // 1000 if note_data.get("time") else None,
            url=url,
        )

    @staticmethod
    def _extract_initial_state_json(html: str) -> dict[str, Any]:
        matched = re.search(r"window\.__INITIAL_STATE__=(.*?)</script>", html)
        if not matched:
            raise ParseException("xhs initial state missing")
        return json.loads(matched.group(1).replace("undefined", "null"))

    @staticmethod
    def _extract_video_url(video_data: Any) -> str | None:
        if not isinstance(video_data, dict):
            return None
        try:
            return convert(video_data, type=Video).video_url
        except Exception:
            return None


class Stream(Struct):
    h264: list[dict[str, Any]] | None = None
    h265: list[dict[str, Any]] | None = None
    av1: list[dict[str, Any]] | None = None
    h266: list[dict[str, Any]] | None = None


class Media(Struct):
    stream: Stream


class Video(Struct):
    media: Media

    @property
    def video_url(self) -> str | None:
        stream = self.media.stream
        if stream.h265:
            return stream.h265[0]["masterUrl"]
        if stream.h264:
            return stream.h264[0]["masterUrl"]
        if stream.av1:
            return stream.av1[0]["masterUrl"]
        if stream.h266:
            return stream.h266[0]["masterUrl"]
        return None