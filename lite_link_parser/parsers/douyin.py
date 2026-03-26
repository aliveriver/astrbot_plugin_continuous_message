from __future__ import annotations

import re
from random import choice
from typing import Any, ClassVar

import msgspec
from msgspec import Struct, field

from astrbot.api import logger

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform
from ..exception import ParseException


class DouyinLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="douyin", display_name="抖音")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        cookie_dir = self.ensure_cookie_dir(self.config["cookie_dir"])
        self.cookiejar = CookieJar(
            cookie_dir,
            name="douyin",
            domain="douyin.com",
            raw_cookies=self.site_config.get("cookies", ""),
        )
        self._set_cookies()

    def _set_cookies(self, cookies_str: str = "") -> None:
        value = cookies_str or self.cookiejar.cookies_str
        if value:
            self.ios_headers["Cookie"] = value
            self.android_headers["Cookie"] = value

    @handle("v.douyin", r"v\.douyin\.com/[a-zA-Z0-9_\-]+")
    @handle("jx.douyin", r"jx\.douyin\.com/[a-zA-Z0-9_\-]+")
    async def _parse_short_link(self, searched: re.Match[str]):
        return await self.parse_with_redirect(f"https://{searched.group(0)}")

    @handle("douyin", r"douyin\.com/(?P<ty>video|note)/(?P<vid>\d+)")
    @handle("iesdouyin", r"iesdouyin\.com/share/(?P<ty>slides|video|note)/(?P<vid>\d+)")
    @handle("m.douyin", r"m\.douyin\.com/share/(?P<ty>slides|video|note)/(?P<vid>\d+)")
    @handle("jingxuan.douyin", r"jingxuan\.douyin\.com/m/(?P<ty>slides|video|note)/(?P<vid>\d+)")
    async def _parse_douyin(self, searched: re.Match[str]):
        ty, vid = searched.group("ty"), searched.group("vid")
        if ty == "slides":
            return await self.parse_slides(vid)

        urls = (
            f"https://m.douyin.com/share/{ty}/{vid}",
            f"https://www.iesdouyin.com/share/{ty}/{vid}",
        )
        for url in urls:
            try:
                return await self.parse_video(url)
            except ParseException as exc:
                logger.warning(f"[link_parser:douyin] parse failed, fallback next url: {exc}")
        raise ParseException("douyin parse failed")

    async def parse_with_redirect(self, url: str):
        async with self.session.get(
            url,
            headers=self.ios_headers,
            allow_redirects=False,
            proxy=self.proxy,
            ssl=False,
        ) as response:
            set_cookie_headers = response.headers.getall("Set-Cookie", [])
            self.cookiejar.update_from_response(set_cookie_headers)
            self._set_cookies()
            redirect_url = response.headers.get("Location", url) if response.status in (301, 302, 303, 307, 308) else url

        if redirect_url == url:
            raise ParseException(f"douyin redirect failed: {url}")
        keyword, searched = self.search_url(redirect_url)
        return await self.parse(keyword, searched)

    async def parse_video(self, url: str):
        async with self.session.get(
            url,
            headers=self.ios_headers,
            allow_redirects=False,
            proxy=self.proxy,
            ssl=False,
        ) as response:
            if response.status != 200:
                raise ParseException(f"douyin status={response.status}")
            text = await response.text()
            set_cookie_headers = response.headers.getall("Set-Cookie", [])
            self.cookiejar.update_from_response(set_cookie_headers)
            self._set_cookies()

        matched = re.search(r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", text, flags=re.DOTALL)
        if not matched or not matched.group(1):
            raise ParseException("douyin router data missing")

        video_data = msgspec.json.decode(matched.group(1).strip(), type=RouterData).video_data
        contents = []
        if video_data.image_urls:
            contents.extend(self.create_image_contents(video_data.image_urls))
        elif video_data.video_url:
            duration = video_data.video.duration if video_data.video else 0
            contents.append(self.create_video_content(video_data.video_url, video_data.cover_url, duration))

        return self.result(
            title=video_data.desc,
            author=self.create_author(video_data.author.nickname, video_data.avatar_url),
            contents=contents,
            timestamp=video_data.create_time,
            url=url,
        )

    async def parse_slides(self, video_id: str):
        async with self.session.get(
            "https://www.iesdouyin.com/web/api/v2/aweme/slidesinfo/",
            params={"aweme_ids": f"[{video_id}]", "request_source": "200"},
            headers=self.android_headers,
            proxy=self.proxy,
            ssl=False,
        ) as response:
            response.raise_for_status()
            set_cookie_headers = response.headers.getall("Set-Cookie", [])
            self.cookiejar.update_from_response(set_cookie_headers)
            self._set_cookies()
            slides = msgspec.json.decode(await response.read(), type=SlidesInfo).aweme_details[0]

        contents = []
        if slides.image_urls:
            contents.extend(self.create_image_contents(slides.image_urls))
        for dynamic_url in slides.dynamic_urls:
            contents.append(self.create_video_content(dynamic_url))

        return self.result(
            title=slides.desc,
            author=self.create_author(slides.name, slides.avatar_url),
            contents=contents,
            timestamp=slides.create_time,
            url=f"https://www.iesdouyin.com/share/slides/{video_id}",
        )


class Avatar(Struct):
    url_list: list[str]


class AuthorModel(Struct):
    nickname: str
    avatar_thumb: Avatar | None = None
    avatar_medium: Avatar | None = None


class PlayAddr(Struct):
    url_list: list[str]


class Cover(Struct):
    url_list: list[str]


class VideoModel(Struct):
    play_addr: PlayAddr
    cover: Cover
    duration: int


class ImageModel(Struct):
    video: VideoModel | None = None
    url_list: list[str] = field(default_factory=list)


class VideoData(Struct):
    create_time: int
    author: AuthorModel
    desc: str
    images: list[ImageModel] | None = None
    video: VideoModel | None = None

    @property
    def image_urls(self) -> list[str]:
        return [choice(image.url_list) for image in self.images] if self.images else []

    @property
    def video_url(self) -> str | None:
        return choice(self.video.play_addr.url_list).replace("playwm", "play") if self.video else None

    @property
    def cover_url(self) -> str | None:
        return choice(self.video.cover.url_list) if self.video else None

    @property
    def avatar_url(self) -> str | None:
        if self.author.avatar_thumb:
            return choice(self.author.avatar_thumb.url_list)
        if self.author.avatar_medium:
            return choice(self.author.avatar_medium.url_list)
        return None


class VideoInfoRes(Struct):
    item_list: list[VideoData] = field(default_factory=list)

    @property
    def video_data(self) -> VideoData:
        if not self.item_list:
            raise ParseException("douyin item_list empty")
        return choice(self.item_list)


class VideoOrNotePage(Struct):
    video_info_res: VideoInfoRes = field(name="videoInfoRes", default_factory=VideoInfoRes)


class LoaderData(Struct):
    video_page: VideoOrNotePage | None = field(name="video_(id)/page", default=None)
    note_page: VideoOrNotePage | None = field(name="note_(id)/page", default=None)


class RouterData(Struct):
    loader_data: LoaderData = field(name="loaderData", default_factory=LoaderData)
    errors: dict[str, Any] | None = None

    @property
    def video_data(self) -> VideoData:
        if self.loader_data.video_page:
            return self.loader_data.video_page.video_info_res.video_data
        if self.loader_data.note_page:
            return self.loader_data.note_page.video_info_res.video_data
        raise ParseException("douyin page data missing")


class SlidesVideo(Struct):
    play_addr: PlayAddr
    cover: Cover
    duration: int


class SlidesImage(Struct):
    video: SlidesVideo | None = None
    url_list: list[str] = field(default_factory=list)


class SlidesAuthor(Struct):
    nickname: str
    avatar_thumb: Avatar


class SlidesData(Struct):
    author: SlidesAuthor
    desc: str
    create_time: int
    images: list[SlidesImage]

    @property
    def name(self) -> str:
        return self.author.nickname

    @property
    def avatar_url(self) -> str:
        return choice(self.author.avatar_thumb.url_list)

    @property
    def image_urls(self) -> list[str]:
        return [choice(image.url_list) for image in self.images]

    @property
    def dynamic_urls(self) -> list[str]:
        return [choice(image.video.play_addr.url_list) for image in self.images if image.video]


class SlidesInfo(Struct):
    aweme_details: list[SlidesData] = field(default_factory=list)
