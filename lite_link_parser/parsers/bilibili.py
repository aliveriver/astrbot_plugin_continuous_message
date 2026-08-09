from __future__ import annotations

import re
from http.cookies import SimpleCookie
from typing import ClassVar

from bilibili_api import Credential, request_settings, select_client
from bilibili_api.video import Video

from ..base import BaseLiteParser, handle
from ..data import Platform

select_client("curl_cffi")
request_settings.set("impersonate", "chrome131")


class BilibiliLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="bilibili", display_name="B站")

    def __init__(self, config: dict):
        super().__init__(config)
        self.headers.update(
            {
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
            }
        )
        self.credential = self._build_credential(self.site_config.get("cookies", ""))

    @handle("b23.tv", r"b23\.tv/[A-Za-z\d\._?%&+\-=/#]+")
    @handle("bili2233", r"bili2233\.cn/[A-Za-z\d\._?%&+\-=/#]+")
    async def _parse_short_link(self, searched: re.Match[str]):
        return await self.parse_with_redirect(f"https://{searched.group(0)}", self.headers)

    @handle("BV", r"(?:^|[\s(（])(?P<bvid>BV[0-9A-Za-z]{10})(?:\s*(?P<page_num>\d{1,3}))?")
    @handle("/BV", r"bilibili\.com(?:/video)?/(?P<bvid>BV[0-9A-Za-z]{10})(?:\?p=(?P<page_num>\d{1,3}))?")
    async def _parse_bv(self, searched: re.Match[str]):
        return await self.parse_video(bvid=str(searched.group("bvid")), page_num=int(searched.group("page_num") or 1))

    @handle("av", r"(?:^|[\s(（])av(?P<avid>\d{6,})(?:\s*(?P<page_num>\d{1,3}))?")
    @handle("/av", r"bilibili\.com(?:/video)?/av(?P<avid>\d{6,})(?:\?p=(?P<page_num>\d{1,3}))?")
    async def _parse_av(self, searched: re.Match[str]):
        return await self.parse_video(avid=int(searched.group("avid")), page_num=int(searched.group("page_num") or 1))

    async def parse_video(
        self,
        *,
        bvid: str | None = None,
        avid: int | None = None,
        page_num: int = 1,
    ):
        kwargs = {"credential": self.credential}
        if bvid:
            kwargs["bvid"] = bvid
        if avid:
            kwargs["aid"] = avid
        video = Video(**kwargs)
        info = await video.get_info()
        title = info.get("title") or "B站视频"
        desc = (info.get("desc") or "").strip() or None
        owner = info.get("owner") or {}
        stat = info.get("stat") or {}
        pic = info.get("pic")
        pages = info.get("pages") or []
        timestamp = info.get("pubdate") or info.get("ctime")

        page_index = max(page_num - 1, 0)
        if len(pages) > 1:
            page = pages[page_index % len(pages)]
            part = (page.get("part") or "").strip()
            if part:
                title = f"{title} - {part}"
            timestamp = page.get("ctime") or timestamp

        stats = []
        mapping = [
            ("播放", stat.get("view")),
            ("点赞", stat.get("like")),
            ("评论", stat.get("reply")),
            ("收藏", stat.get("favorite")),
            ("投币", stat.get("coin")),
            ("分享", stat.get("share")),
            ("弹幕", stat.get("danmaku")),
        ]
        for label, value in mapping:
            if isinstance(value, int):
                stats.append(f"{label} {value}")

        url = f"https://www.bilibili.com/video/{info.get('bvid') or bvid}" if (info.get("bvid") or bvid) else None
        if url and len(pages) > 1 and page_num > 1:
            url += f"?p={page_num}"

        contents = self.create_image_contents([pic] if pic else [])
        return self.result(
            title=title,
            text=desc,
            author=self.create_author(owner.get("name"), owner.get("face")),
            contents=contents,
            timestamp=timestamp,
            url=url,
            extra_info=" | ".join(stats) if stats else None,
        )

    @staticmethod
    def _build_credential(cookie_string: str) -> Credential | None:
        clean = cookie_string.strip()
        if not clean:
            return None
        parsed = SimpleCookie()
        parsed.load(clean)
        kwargs = {}
        mapping = {
            "SESSDATA": "sessdata",
            "bili_jct": "bili_jct",
            "BUVID3": "buvid3",
            "DedeUserID": "dedeuserid",
            "ac_time_value": "ac_time_value",
        }
        for cookie_name, field_name in mapping.items():
            morsel = parsed.get(cookie_name)
            if morsel and morsel.value:
                kwargs[field_name] = morsel.value
        return Credential(**kwargs) if kwargs else None
