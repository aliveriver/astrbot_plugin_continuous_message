from __future__ import annotations

import json
from typing import ClassVar

from aiohttp import ClientError

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform
from ..exception import ParseException


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

        # 1. 获取歌曲详情
        async with self.session.get(detail_url, headers=self.headers, proxy=self.proxy) as response:
            if response.status >= 400:
                raise ClientError(f"ncm detail failed {response.status}")
            detail_json = json.loads(await response.text())

        songs = detail_json.get("songs") or []
        if not songs:
            raise ParseException("未获取到网易云歌曲详情或歌曲下架")
        
        song = songs[0]
        title = song.get("name", "")
        
        # 安全获取别名（修复越界漏洞 1）
        alias_list = song.get("alias") or []
        sub_title = alias_list[0] if alias_list else ""

        # 安全获取专辑与封面
        album = song.get("album") or {}
        album_name = album.get("name", "")
        pic_url = album.get("picUrl", "")
        cover_url = f"{pic_url}?param=640y640" if pic_url else ""
        
        duration_ms = song.get("duration", 0)
        
        # 安全获取歌手信息
        artists = song.get("artists") or []
        author_name = " / ".join(item.get("name", "") for item in artists if item.get("name")) or "未知歌手"
        author_avatar = artists[0].get("img1v1Url", "") if artists else ""

        # 2. 获取播放直链
        audio_url = ""
        try:
            async with self.session.get(play_url, headers=self.headers, proxy=self.proxy) as response:
                if response.status < 400:
                    play_json = json.loads(await response.text())
                    data_list = play_json.get("data") or []
                    if data_list:
                        audio_url = data_list[0].get("url") or ""
        except Exception:
            pass

        # 组装播放组件
        contents = []
        if audio_url:
            audio = self.create_audio_content(
                audio_url,
                cover_url=cover_url,
                duration=duration_ms // 1000,
                name=title,
            )
            contents.append(audio)

        display_title = f"{title}（{sub_title}）" if sub_title else title
        return self.result(
            title=display_title or "网易云音乐",
            text=f"专辑：{album_name}" if album_name else "单曲",
            author=self.create_author(author_name, author_avatar),
            contents=contents,
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