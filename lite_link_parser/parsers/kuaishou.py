from __future__ import annotations

import re
from random import choice
from typing import Any, ClassVar, TypeAlias

import msgspec
from msgspec import Struct, field

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform
from ..exception import ParseException


class KuaishouLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="kuaishou", display_name="快手")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.ios_headers.update({"Referer": "https://v.kuaishou.com/"})
        cookie_dir = self.ensure_cookie_dir(self.config["cookie_dir"])
        self.cookiejar = CookieJar(
            cookie_dir,
            name="kuaishou",
            domain="kuaishou.com",
            raw_cookies=self.site_config.get("cookies", ""),
        )
        if self.cookiejar.cookies_str:
            self.ios_headers["cookie"] = self.cookiejar.cookies_str

    @handle("v.kuaishou", r"v\.kuaishou\.com/[A-Za-z\d._?%&+\-=/#]+")
    @handle("kuaishou", r"(?:www\.)?kuaishou\.com/[A-Za-z\d._?%&+\-=/#]+")
    @handle("chenzhongtech", r"(?:v\.m\.)?chenzhongtech\.com/fw/[A-Za-z\d._?%&+\-=/#]+")
    async def _parse(self, searched: re.Match[str]):
        url = f"https://{searched.group(0)}"
        real_url = await self.get_redirect_url(url, headers=self.ios_headers)
        if not real_url:
            raise ParseException("kuaishou redirect failed")
        real_url = real_url.replace("/fw/long-video/", "/fw/photo/")

        async with self.session.get(real_url, headers=self.ios_headers, proxy=self.proxy) as response:
            if response.status >= 400:
                raise ParseException(f"kuaishou status={response.status}")
            html = await response.text()

        matched = re.search(r"window\.INIT_STATE\s*=\s*(.*?)</script>", html)
        if not matched:
            raise ParseException("kuaishou init_state missing")
        init_state = msgspec.json.decode(matched.group(1).strip(), type=KuaishouInitState)
        photo = next((item.photo for item in init_state.values() if item.photo is not None), None)
        if photo is None:
            raise ParseException("kuaishou photo missing")

        contents = []
        if photo.video_url:
            contents.append(self.create_video_content(photo.video_url, photo.cover_url, photo.duration))
        if photo.img_urls:
            contents.extend(self.create_image_contents(photo.img_urls))

        return self.result(
            title=photo.caption,
            author=self.create_author(photo.name, photo.head_url),
            contents=contents,
            timestamp=photo.timestamp // 1000,
            url=real_url,
        )


class CdnUrl(Struct):
    cdn: str
    url: str | None = None


class Atlas(Struct):
    cdn_list: list[CdnUrl] = field(name="cdnList", default_factory=list)
    img_route_list: list[str] = field(name="list", default_factory=list)

    @property
    def img_urls(self) -> list[str]:
        if not self.cdn_list or not self.img_route_list:
            return []
        cdn = choice(self.cdn_list).cdn
        return [f"https://{cdn}/{url}" for url in self.img_route_list]


class ExtParams(Struct):
    atlas: Atlas = field(default_factory=Atlas)


class Photo(Struct):
    caption: str
    timestamp: int
    duration: int = 0
    user_name: str = field(default="未知用户", name="userName")
    head_url: str | None = field(default=None, name="headUrl")
    cover_urls: list[CdnUrl] = field(name="coverUrls", default_factory=list)
    main_mv_urls: list[CdnUrl] = field(name="mainMvUrls", default_factory=list)
    ext_params: ExtParams = field(name="ext_params", default_factory=ExtParams)

    @property
    def name(self) -> str:
        return self.user_name.replace("\u3164", "").strip()

    @property
    def cover_url(self) -> str | None:
        return choice(self.cover_urls).url if self.cover_urls else None

    @property
    def video_url(self) -> str | None:
        return choice(self.main_mv_urls).url if self.main_mv_urls else None

    @property
    def img_urls(self) -> list[str]:
        return self.ext_params.atlas.img_urls


class TusjohData(Struct):
    result: int
    photo: Photo | None = None


KuaishouInitState: TypeAlias = dict[str, TusjohData]
