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
            "Referer": "https://y.qq.com/",
            "Origin": "https://y.qq.com"
        })
        
        cookie_dir = self.ensure_cookie_dir(self.config.get("cookie_dir", "data/cookies"))
        cookie_file = cookie_dir / "qqmusic_cookies.txt"
        if cookie_file.exists():
            try:
                with open(cookie_file, "r", encoding="utf-8") as f:
                    self.headers["Cookie"] = f.read().strip()
            except Exception:
                pass

    # 1. 匹配 .m4a / .mp3 音频直链（修复 no parser matched 报错）
    @handle("stream.qqmusic.qq.com", r"https?://[^/]*stream\.qqmusic\.qq\.com/.*\.m4a(?:\?.*)?$")
    @handle("stream.qqmusic.qq.com", r"https?://[^/]*stream\.qqmusic\.qq\.com/.*\.mp3(?:\?.*)?$")
    async def _parse_direct_audio(self, searched: re.Match[str]):
        url = searched.group(0)
        return self.result(
            title="QQ音乐",
            text="音频流链接",
            contents=[self.create_audio_content(url)],
            url=url,
        )

    # 2. 处理 QQ 音乐短链接重定向
    @handle("c6.y.qq.com", r"c6\.y\.qq\.com/[A-Za-z0-9._?%&+=/#@-]+")
    async def _parse_short_link(self, searched: re.Match[str]):
        return await self.parse_with_redirect(f"https://{searched.group(0)}", self.headers)

    # 3. 核心匹配：支持 i.y.qq.com, y.qq.com, c.y.qq.com 等多种卡片/详情页链接
    @handle("i.y.qq.com", r"i\.y\.qq\.com/v8/playsong\.html\?[^>\s]*songmid=(?P<mid>[A-Za-z0-9]+)")
    @handle("c.y.qq.com", r"c\.y\.qq\.com/[^>\s]*songmid=(?P<mid>[A-Za-z0-9]+)")
    @handle("y.qq.com/n/ryqq/songDetail", r"y\.qq\.com/n/ryqq/songDetail/(?P<mid>[A-Za-z0-9]+)")
    @handle("y.qq.com/n/yqq/song", r"y\.qq\.com/n/yqq/song/(?P<mid>[A-Za-z0-9]+)\.html")
    async def _parse_song(self, searched: re.Match[str]):
        mid = searched.group("mid")
        api_url = f"https://u.y.qq.com/cgi-bin/musicu.fcg?data=%7B%22songinfo%22%3A%7B%22method%22%3A%22get_song_detail_yqq%22%2C%22module%22%3A%22music.pf_song_detail_svr%22%2C%22param%22%3A%7B%22song_mid%22%3A%22{mid}%22%7D%7D%7D"

        try:
            async with curl_requests.AsyncSession(impersonate="chrome110") as session:
                resp = await session.get(api_url, headers=self.headers, timeout=10)
                data = resp.json()

            songinfo_data = data.get("songinfo", {}).get("data", {})
            track_info = songinfo_data.get("track_info")
            if not track_info:
                raise ParseException("QQ音乐接口未返回歌曲详情，可能是MID已失效或版权受限")

            title = track_info.get("name", "未知歌曲")
            album = track_info.get("album", {}).get("name", "未知专辑")
            singers = [s.get("name", "") for s in track_info.get("singer", []) if s.get("name")]
            author_name = " / ".join(singers) or "未知歌手"
            
            album_mid = track_info.get("album", {}).get("mid", "")
            cover_url = f"https://y.gtimg.cn/music/photo_new/T002R300x300M000{album_mid}.jpg" if album_mid else None
            
            audio_url = f"https://i.y.qq.com/v8/playsong.html?songmid={mid}&ADTAG=myqq&from=myqq&channel=10007100"

            lyrics_text = await self._fetch_lyrics(mid)
            # 限制歌词预览保留最大 400 字，避免提示词过长
            if len(lyrics_text) > 400:
                lyrics_text = lyrics_text[:400].rstrip() + "\n..."

            display_text = f"专辑：{album}\n\n【歌词预览】\n{lyrics_text}"

            return self.result(
                title=title,
                text=display_text,
                author=self.create_author(name=author_name),
                contents=[self.create_audio_content(audio_url, cover_url=cover_url, name=title)],
                url=f"https://y.qq.com/n/ryqq/songDetail/{mid}",
            )
        except Exception as e:
            if isinstance(e, ParseException):
                raise e
            raise ParseException(f"QQ音乐数据解析失败: {e}")

    async def _fetch_lyrics(self, mid: str) -> str:
        lyric_api = f"https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg?songmid={mid}&format=json&nobase64=1"
        headers = self.headers.copy()
        headers["Referer"] = "https://y.qq.com/n/ryqq/player"
        
        try:
            async with curl_requests.AsyncSession(impersonate="chrome110") as session:
                resp = await session.get(lyric_api, headers=headers, timeout=8)
                text = resp.text.strip()
                
                # 使用正则通用安全提取 JSON 部分（兼容 JSONP 变体）
                json_match = re.search(r"(\{.*\})", text)
                if json_match:
                    text = json_match.group(1)
                
                data = json.loads(text)
                lyric = data.get("lyric", "")
                if lyric:
                    # 清洗时间轴 [00:00.00] 和 [ar:xxx] 标签
                    clean_l = re.sub(r'\[\d{2,}:\d{2}(?:[:\.]\d{1,3})?\]', '', lyric).strip()
                    clean_l = re.sub(r'\[[a-zA-Z]+:[^\]]*\]', '', clean_l).strip()
                    clean_l = re.sub(r'\n+', '\n', clean_l)
                    return clean_l.strip() or "（暂无歌词内容）"
        except Exception:
            pass
        return "（未能获取到歌词）"