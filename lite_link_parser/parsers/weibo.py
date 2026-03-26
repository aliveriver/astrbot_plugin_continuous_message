from __future__ import annotations

from email.utils import parsedate_to_datetime
from re import Match, sub
from time import time
from typing import ClassVar
from uuid import uuid4

import msgspec
from aiohttp import ClientError
from bs4 import BeautifulSoup, Tag
from msgspec import Struct

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import ParseResult, Platform
from ..exception import ParseException


class WeiboLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="weibo", display_name="微博")

    def __init__(self, config: dict):
        super().__init__(config)
        self.headers.update(
            {
                "accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                    "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
                ),
                "referer": "https://weibo.com/",
            }
        )
        cookie_dir = self.ensure_cookie_dir(self.config["cookie_dir"])
        self.cookiejar = CookieJar(
            cookie_dir,
            name="weibo",
            domain="weibo.com",
            raw_cookies=self.site_config.get("cookies", ""),
        )
        if self.cookiejar.cookies_str:
            self.headers["cookie"] = self.cookiejar.cookies_str

    @handle("weibo.com/tv", r"weibo\.com/tv/show/\d{4}:\d+\?mid=(?P<mid>\d+)")
    async def _parse_weibo_tv(self, searched: Match[str]):
        return await self.parse_weibo_id(self._mid2id(str(searched.group("mid"))))

    @handle("video.weibo", r"video\.weibo\.com/show\?fid=(?P<fid>\d+:\d+)")
    async def _parse_video_weibo(self, searched: Match[str]):
        return await self.parse_fid(str(searched.group("fid")))

    @handle("m.weibo.cn", r"weibo\.cn/(?:status|detail|\d+)/(?P<wid>[0-9a-zA-Z]+)")
    @handle("weibo.com", r"weibo\.com/\d+/(?P<wid>[0-9a-zA-Z]+)")
    async def _parse_status(self, searched: Match[str]):
        return await self.parse_weibo_id(str(searched.group("wid")))

    @handle("mapp.api.weibo", r"mapp\.api\.weibo\.cn/fx/[A-Za-z\d]+\.html")
    async def _parse_mapp(self, searched: Match[str]):
        return await self.parse_with_redirect(f"https://{searched.group(0)}")

    @handle("weibo.com/ttarticle", r"id=(?P<id>\d+)")
    @handle("weibo.com/article", r"/id/(?P<id>\d+)")
    async def _parse_article(self, searched: Match[str]):
        return await self.parse_article(str(searched.group("id")))

    async def parse_article(self, article_id: str):
        class UserInfo(Struct):
            screen_name: str
            profile_image_url: str

        class Data(Struct):
            url: str
            title: str
            content: str
            userinfo: UserInfo
            create_at_unix: int

        class Detail(Struct):
            code: str
            msg: str
            data: Data

        async with self.session.post(
            "https://card.weibo.com/article/m/aj/detail",
            data={"_rid": str(uuid4()), "id": article_id, "_t": int(time() * 1000)},
            headers=self.headers,
            proxy=self.proxy,
        ) as response:
            if response.status >= 400:
                raise ClientError(f"weibo article api {response.status} {response.reason}")
            detail = msgspec.json.decode(await response.read(), type=Detail)

        if detail.msg != "success":
            raise ParseException("weibo article request failed")

        soup = BeautifulSoup(detail.data.content, "html.parser")
        contents = []
        text_buffer: list[str] = []
        for element in soup.find_all(["p", "img"]):
            if not isinstance(element, Tag):
                continue
            if element.name == "p":
                text = element.get_text(strip=True).replace("\u200b", "")
                if text:
                    text_buffer.append(text)
            elif element.name == "img":
                src = element.get("src")
                if isinstance(src, str):
                    contents.append(self.create_graphics_content(src, text="\n\n".join(text_buffer)))
                    text_buffer.clear()

        return self.result(
            url=detail.data.url,
            title=detail.data.title,
            author=self.create_author(
                detail.data.userinfo.screen_name,
                detail.data.userinfo.profile_image_url,
            ),
            timestamp=detail.data.create_at_unix,
            text="\n\n".join(text_buffer) if text_buffer else None,
            contents=contents,
        )

    async def parse_fid(self, fid: str):
        async with self.session.post(
            f"https://h5.video.weibo.com/api/component?page=/show/{fid}",
            data='data={"Component_Play_Playinfo":{"oid":"' + fid + '"}}',
            headers={
                "Referer": f"https://h5.video.weibo.com/show/{fid}",
                "Content-Type": "application/x-www-form-urlencoded",
                **self.headers,
            },
            proxy=self.proxy,
        ) as response:
            if response.status >= 400:
                raise ClientError(f"weibo video api {response.status} {response.reason}")
            payload = await response.json()

        data = payload.get("data", {}).get("Component_Play_Playinfo", {})
        if not data:
            raise ParseException("weibo video info missing")

        user = data.get("reward", {}).get("user", {})
        text = sub(r"<[^>]*>", "", data.get("text", "")).replace("\n\n", "").strip() or None
        cover_url = data.get("cover_image")
        if cover_url:
            cover_url = "https:" + cover_url

        contents = []
        video_url_dict = data.get("urls")
        if video_url_dict and isinstance(video_url_dict, dict):
            video_url = "https:" + next(iter(video_url_dict.values()))
        else:
            video_url = data.get("stream_url")
        if video_url:
            contents.append(self.create_video_content(video_url, cover_url))

        return self.result(
            title=data.get("title") or None,
            text=text,
            author=self.create_author(
                user.get("name", "未知"),
                user.get("profile_image_url"),
                user.get("description"),
            ),
            contents=contents,
            timestamp=data.get("real_date"),
        )

    async def parse_weibo_id(self, weibo_id: str):
        headers = {
            "accept": "application/json, text/plain, */*",
            "referer": f"https://m.weibo.cn/detail/{weibo_id}",
            "origin": "https://m.weibo.cn",
            "x-requested-with": "XMLHttpRequest",
            "mweibo-pwa": "1",
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            **self.headers,
        }
        async with self.session.get(
            f"https://m.weibo.cn/statuses/show?id={weibo_id}&_={int(time() * 1000)}",
            headers=headers,
            allow_redirects=False,
            proxy=self.proxy,
        ) as response:
            if response.status != 200:
                raise ParseException(f"weibo status request failed: {response.status}")
            if "application/json" not in response.headers.get("content-type", ""):
                raise ParseException("weibo response is not json")
            data = msgspec.json.decode(await response.read(), type=WeiboResponse).data
        return self.build_weibo_data(data)

    def build_weibo_data(self, data: "WeiboData") -> ParseResult:
        contents = []
        if data.video_url:
            contents.append(self.create_video_content(data.video_url, data.cover_url))
        if data.image_urls:
            contents.extend(self.create_image_contents(data.image_urls))

        return self.result(
            title=data.title,
            text=data.text_content,
            author=self.create_author(data.display_name, data.user.profile_image_url),
            contents=contents,
            timestamp=data.timestamp,
            url=data.url,
            repost=self.build_weibo_data(data.retweeted_status) if data.retweeted_status else None,
        )

    @staticmethod
    def _base62_encode(number: int) -> str:
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if number == 0:
            return "0"
        result = ""
        while number > 0:
            result = alphabet[number % 62] + result
            number //= 62
        return result

    def _mid2id(self, mid: str) -> str:
        from math import ceil

        mid = str(mid)[::-1]
        size = ceil(len(mid) / 7)
        result = []
        for index in range(size):
            chunk = mid[index * 7 : (index + 1) * 7][::-1]
            encoded = self._base62_encode(int(chunk))
            if index < size - 1 and len(encoded) < 4:
                encoded = "0" * (4 - len(encoded)) + encoded
            result.append(encoded)
        result.reverse()
        return "".join(result)


class LargeInPic(Struct):
    url: str


class Pic(Struct):
    url: str
    large: LargeInPic


class Urls(Struct):
    mp4_720p_mp4: str | None = None
    mp4_hd_mp4: str | None = None
    mp4_ld_mp4: str | None = None

    def get_video_url(self) -> str | None:
        return self.mp4_720p_mp4 or self.mp4_hd_mp4 or self.mp4_ld_mp4 or None


class PagePic(Struct):
    url: str


class PageInfo(Struct):
    title: str | None = None
    urls: Urls | None = None
    page_pic: PagePic | None = None


class User(Struct):
    id: int
    screen_name: str
    profile_image_url: str


class WeiboData(Struct):
    user: User
    text: str
    bid: str
    created_at: str
    status_title: str | None = None
    pics: list[Pic] | None = None
    page_info: PageInfo | None = None
    retweeted_status: "WeiboData | None" = None

    @property
    def title(self) -> str | None:
        return self.page_info.title if self.page_info else None

    @property
    def display_name(self) -> str:
        return self.user.screen_name

    @property
    def text_content(self) -> str:
        return sub(r"<[^>]*>", "", self.text.replace("<br />", "\n"))

    @property
    def cover_url(self) -> str | None:
        return self.page_info.page_pic.url if self.page_info and self.page_info.page_pic else None

    @property
    def video_url(self) -> str | None:
        return self.page_info.urls.get_video_url() if self.page_info and self.page_info.urls else None

    @property
    def image_urls(self) -> list[str]:
        return [item.large.url for item in self.pics] if self.pics else []

    @property
    def url(self) -> str:
        return f"https://weibo.com/{self.user.id}/{self.bid}"

    @property
    def timestamp(self) -> int:
        return int(parsedate_to_datetime(self.created_at).timestamp())


class WeiboResponse(Struct):
    ok: int
    data: WeiboData
