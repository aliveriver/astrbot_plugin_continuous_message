from __future__ import annotations

import json
from typing import ClassVar

from aiohttp import ClientError

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform


class NCMLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="ncm", display_name="网易云音乐")

    def __init__(self, config: dict):
        super().__init__(config)
        self.headers.update({"Referer": "https://music.163.com"})
        cookie_dir = self.ensure_cookie_dir(self.config["cookie_dir"])
        self.cookiejar = CookieJar(
            cookie_dir,
            name="ncm",
            domain="music.163.com",
            raw_cookies=self.site_config.get("cookies", ""),
        )
        if self.cookiejar.cookies_str:
            self.headers["cookie"] = self.cookiejar.cookies_str

    @handle("163cn.tv", r"163cn\.tv/(?P<short_key>\w+)")
    async def _parse_short(self, searched):
        return await self.parse_with_redirect(f"https://163cn.tv/{searched.group('short_key')}")

    @handle("y.music.163.com", r"y\.music\.163\.com/m/song\?.*id=(?P<song_id>\d+)")
    @handle("music.163.com/song", r"music\.163\.com/song/?\?.*id=(?P<song_id>\d+)")
    @handle("music.163.com/#/song", r"music\.163\.com/#/song\?.*id=(?P<song_id>\d+)")
    async def _parse_song(self, searched):
        song_id = searched.group("song_id")
        detail_url = f"https://music.163.com/api/song/detail/?id={song_id}&ids=[{song_id}]"
        play_url = f"https://music.163.com/api/song/enhance/player/url?ids=[{song_id}]&br=320000"

        async with self.session.get(detail_url, headers=self.headers, proxy=self.proxy) as response:
            if response.status >= 400:
                raise ClientError(f"ncm detail failed {response.status}")
            detail_json = json.loads(await response.text())

        song = detail_json.get("songs", [{}])[0]
        title = song.get("name", "")
        sub_title = song.get("alias", [""])[0]
        album_name = song.get("album", {}).get("name", "")
        cover_url = song.get("album", {}).get("picUrl", "") + "?param=640y640"
        duration_ms = song.get("duration", 0)
        artists = song.get("artists", [])
        author_name = " / ".join(item.get("name", "") for item in artists)
        author_avatar = artists[0].get("img1v1Url", "") if artists else ""

        async with self.session.get(play_url, headers=self.headers, proxy=self.proxy) as response:
            if response.status >= 400:
                raise ClientError(f"ncm play failed {response.status}")
            play_json = json.loads(await response.text())

        play_info = play_json.get("data", [{}])[0]
        audio_url = play_info.get("url", "")
        audio = self.create_audio_content(
            audio_url,
            cover_url=cover_url,
            duration=duration_ms // 1000,
            name=title,
        )
        display_title = f"{title}（{sub_title}）" if sub_title else title
        return self.result(
            title=display_title,
            text=f"专辑：{album_name}",
            author=self.create_author(author_name, author_avatar),
            contents=[audio],
            url=f"https://music.163.com/song?id={song_id}",
        )

    @handle("music.126.net", r"https?://[^/]*music\.126\.net/.*\.mp3(?:\?.*)?$")
    async def _parse_direct_mp3(self, searched):
        url = searched.group(0)
        return self.result(
            title="网易云音乐",
            text="直链音频",
            contents=[self.create_audio_content(url)],
            url=url,
        )

    @handle(
        "music.163.com/song/media/outer/url",
        r"(https?://music\.163\.com/song/media/outer/url\?[^>\s]+)",
    )
    async def _parse_outer(self, searched):
        url = searched.group(0)
        return self.result(
            title="网易云音乐（外链）",
            text="直链音频",
            contents=[self.create_audio_content(url)],
            url=url,
        )
