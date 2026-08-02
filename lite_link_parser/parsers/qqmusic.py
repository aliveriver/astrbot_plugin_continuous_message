from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from curl_cffi import requests as curl_requests
from astrbot.api import logger

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform
from ..exception import ParseException

class QQMusicLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="qqmusic", display_name="QQ音乐")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Referer": "https://y.qq.com/"
        })
        
        cookie_dir = self.ensure_cookie_dir(self.config.get("cookie_dir", "data/cookies"))
        cookie_file = cookie_dir / "qqmusic_cookies.txt"
        if cookie_file.exists():
            try:
                with open(cookie_file, "r", encoding="utf-8") as f:
                    self.headers["Cookie"] = f.read().strip()
            except Exception:
                pass

    # 1. 处理 QQ 音乐短链接
    @handle("c6.y.qq.com", r"c6\.y\.qq\.com/[A-Za-z0-9._?%&+=/#@-]+")
    async def _parse_short_link(self, searched: re.Match[str]):
        return await self.parse_with_redirect(f"https://{searched.group(0)}", self.headers)

    # 2. 【核心新增】处理 QQ 音乐移动端/QQ卡片 链接 (i.y.qq.com)
    @handle("i.y.qq.com", r"i\.y\.qq\.com/v8/playsong\.html\?[^>\s]*songmid=(?P<mid>[A-Za-z0-9]+)")
    # 3. 处理 QQ 音乐标准详情页链接
    @handle("y.qq.com/n/ryqq/songDetail", r"y\.qq\.com/n/ryqq/songDetail/(?P<mid>[A-Za-z0-9]+)")
    @handle("y.qq.com/n/yqq/song", r"y\.qq\.com/n/yqq/song/(?P<mid>[A-Za-z0-9]+)\.html")
    async def _parse_song(self, searched: re.Match[str]):
        mid = searched.group("mid")
        # QQ 音乐歌曲详情 API
        api_url = f"https://u.y.qq.com/cgi-bin/musicu.fcg?data=%7B%22songinfo%22%3A%7B%22method%22%3A%22get_song_detail_yqq%22%2C%22module%22%3A%22music.pf_song_detail_svr%22%2C%22param%22%3A%7B%22song_mid%22%3A%22{mid}%22%7D%7D%7D"

        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            resp = await session.get(api_url, headers=self.headers)
            data = resp.json()

        try:
            # 兼容性检查，防止返回数据结构变化导致崩溃
            songinfo_data = data.get("songinfo", {}).get("data", {})
            track_info = songinfo_data.get("track_info")
            if not track_info:
                raise ParseException("QQ音乐接口未返回歌曲详情，可能是MID已失效")

            title = track_info.get("name", "未知歌曲")
            album = track_info.get("album", {}).get("name", "未知专辑")
            singers = [s.get("name", "") for s in track_info.get("singer", [])]
            author_name = " / ".join(singers)
            
            album_mid = track_info.get("album", {}).get("mid", "")
            cover_url = f"https://y.gtimg.cn/music/photo_new/T002R300x300M000{album_mid}.jpg" if album_mid else None
            
            audio_url = f"https://i.y.qq.com/v8/playsong.html?songmid={mid}&ADTAG=myqq&from=myqq&channel=10007100"

            lyrics_text = await self._fetch_lyrics(mid)
            display_text = f"专辑：{album}\n\n【歌词】\n{lyrics_text}"

            return self.result(
                title=title,
                text=display_text,
                author=self.create_author(name=author_name),
                contents=[self.create_audio_content(audio_url, cover_url=cover_url, name=title)],
                url=f"https://y.qq.com/n/ryqq/songDetail/{mid}",
            )
        except (KeyError, TypeError) as e:
            raise ParseException(f"QQ音乐数据解析失败: {e}")

    async def _fetch_lyrics(self, mid: str) -> str:
        lyric_api = f"https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg?songmid={mid}&format=json&nobase64=1"
        headers = self.headers.copy()
        headers["Referer"] = "https://y.qq.com/n/ryqq/player"
        
        try:
            async with curl_requests.AsyncSession(impersonate="chrome110") as session:
                resp = await session.get(lyric_api, headers=headers)
                text = resp.text.strip()
                if text.startswith("MusicJsonCallback("):
                    text = text[len("MusicJsonCallback("):-1]
                data = json.loads(text)
                
                lyric = data.get("lyric", "")
                if lyric:
                    # 清洗时间轴和标签
                    clean_l = re.sub(r'\[\d{2,}:\d{2}(?:[:\.]\d{1,3})?\]', '', lyric).strip()
                    clean_l = re.sub(r'\[[a-zA-Z]+:[^\]]*\]', '', clean_l).strip()
                    return re.sub(r'\n+', '\n', clean_l) or "（暂无歌词内容）"
        except Exception:
            pass
        return "（未能获取到歌词）"