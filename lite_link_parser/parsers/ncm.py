from __future__ import annotations

import json
import re
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
        # 采用 astrbot_plugin_ncm_get 的极简伪装头和 HTTP 引用源
        self.headers.update({
            "Referer": "http://music.163.com/",
            "User-Agent": "Mozilla/5.0"
        })
        
        cookie_dir = self.ensure_cookie_dir(self.config.get("cookie_dir", "data/cookies"))
        cookie_str = self.site_config.get("cookies", "")
        
        # 保持强力的双重 Cookie 逻辑
        self.cookiejar = CookieJar(
            cookie_dir,
            name="ncm",
            domain="music.163.com",
            raw_cookies=cookie_str,
        )
        
        if self.cookiejar.cookies_str:
            self.headers["cookie"] = self.cookiejar.cookies_str
        else:
            cookie_file = cookie_dir / "ncm_cookies.txt"
            if cookie_file.exists():
                try:
                    with open(cookie_file, "r", encoding="utf-8") as f:
                        raw_text = f.read().strip()
                        if raw_text and not raw_text.startswith("# Netscape"):
                            self.headers["cookie"] = raw_text
                except Exception:
                    pass

    @handle("163cn.tv", r"163cn\.tv/(?P<short_key>\w+)")
    async def _parse_short(self, searched):
        return await self.parse_with_redirect(f"https://163cn.tv/{searched.group('short_key')}")

    @handle("y.music.163.com", r"y\.music\.163\.com/m/song\?.*id=(?P<song_id>\d+)")
    @handle("music.163.com/song", r"music\.163\.com/song/?\?.*id=(?P<song_id>\d+)")
    @handle("music.163.com/#/song", r"music\.163\.com/#/song\?.*id=(?P<song_id>\d+)")
    async def _parse_song(self, searched):
        song_id = searched.group("song_id")
        
        # 参考 ncm_get 插件，使用 http 协议 API
        detail_url = f"http://music.163.com/api/song/detail/?id={song_id}&ids=[{song_id}]"
        play_url = f"http://music.163.com/api/song/enhance/player/url?ids=[{song_id}]&br=320000"
        
        # 歌词接口：lv/kv/tv 使用 -1 以获取最新版本，避免新歌返回空值
        lyric_url = f"http://music.163.com/api/song/lyric?id={song_id}&lv=-1&kv=-1&tv=-1"

        # 1. 获取歌曲基础信息
        async with self.session.get(detail_url, headers=self.headers, proxy=self.proxy) as response:
            if response.status >= 400:
                raise ClientError(f"ncm detail failed {response.status}")
            detail_json = json.loads(await response.text())

        songs = detail_json.get("songs", [])
        if not songs:
            raise ParseException("歌曲信息获取失败，可能是VIP专属、无版权或已下架")
        song = songs[0]

        title = song.get("name", "")
        sub_title = song.get("alias", [""])[0] if song.get("alias") else ""
        album_name = song.get("album", {}).get("name", "")
        cover_url = song.get("album", {}).get("picUrl", "") + "?param=640y640"
        duration_ms = song.get("duration", 0)
        artists = song.get("artists", [])
        author_name = " / ".join(item.get("name", "") for item in artists)
        author_avatar = artists[0].get("img1v1Url", "") if artists else ""

        # 2. 获取歌曲播放直链
        async with self.session.get(play_url, headers=self.headers, proxy=self.proxy) as response:
            if response.status >= 400:
                raise ClientError(f"ncm play failed {response.status}")
            play_json = json.loads(await response.text())

        play_data = play_json.get("data", [])
        if not play_data:
            raise ParseException("音频播放链接获取失败")
        play_info = play_data[0]
        
        audio_url = play_info.get("url", "")
        if not audio_url:
            audio_url = f"https://music.163.com/song/media/outer/url?id={song_id}.mp3"

        audio = self.create_audio_content(
            audio_url,
            cover_url=cover_url,
            duration=duration_ms // 1000,
            name=title,
        )

        # 3. 歌词抓取与清洗 (参考 ncm_get 的正则但增加更严谨的判断)
        lyrics_text = "（未能获取到歌词）"
        try:
            async with self.session.get(lyric_url, headers=self.headers, proxy=self.proxy) as response:
                if response.status == 200:
                    data = json.loads(await response.text())
                    
                    if data.get("nolyric"):
                        lyrics_text = "（纯音乐，无歌词）"
                    elif data.get("uncollected"):
                        lyrics_text = "（网易云暂未收录歌词）"
                    elif 'lrc' in data and 'lyric' in data['lrc']:
                        raw_lyric = data['lrc']['lyric']
                        if raw_lyric and raw_lyric.strip():
                            # 过滤 [by:xxx] 等标签
                            clean_l = re.sub(r'\[[a-zA-Z]+:[^\]]*\]', '', raw_lyric).strip()
                            # 过滤时间轴：支持 [00:00]、[00:00.00]、[00:00.000]
                            clean_l = re.sub(r'\[\d{2,}:\d{2}(?:[:\.]\d{1,3})?\]', '', clean_l).strip()
                            # 合并多余换行
                            lyrics_text = re.sub(r'\n+', '\n', clean_l).strip()
                            
                            if not lyrics_text:
                                lyrics_text = "（暂无有效歌词内容）"
                        else:
                            lyrics_text = "（暂无歌词文本）"
        except Exception:
            pass

        # 拼装返回文本
        display_title = f"{title}（{sub_title}）" if sub_title else title
        display_text = f"专辑：{album_name}\n\n【歌词】\n{lyrics_text}"

        return self.result(
            title=display_title,
            text=display_text,
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