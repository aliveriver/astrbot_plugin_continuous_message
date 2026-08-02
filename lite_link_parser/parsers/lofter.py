from __future__ import annotations
import re
import json
from typing import Any, ClassVar
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from astrbot.api import logger
from ..base import BaseLiteParser, handle
from ..data import Platform
from ..exception import ParseException

class LofterLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="lofter", display_name="LOFTER")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Referer": "https://www.lofter.com/"
        })
        
        cookie_dir = self.ensure_cookie_dir(self.config.get("cookie_dir", "data/cookies"))
        cookie_file = cookie_dir / "lofter_cookies.txt"
        if cookie_file.exists():
            try:
                with open(cookie_file, "r", encoding="utf-8") as f:
                    self.headers["Cookie"] = f.read().strip()
            except Exception:
                pass

    @handle("lofter.com", r"(?P<username>[a-zA-Z0-9-]+)\.lofter\.com/post/(?P<post_id>[a-zA-Z0-9_]+)")
    async def _parse_post(self, searched: re.Match[str]):
        username = searched.group("username")
        post_id = searched.group("post_id")
        url = f"https://{username}.lofter.com/post/{post_id}"

        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            response = await session.get(url, headers=self.headers)
            html_text = response.text

        soup = BeautifulSoup(html_text, "html.parser")
        
        content_div = soup.find("div", class_="content") or soup.find("div", class_="text")
        # 移除 [:500] 限制，交由 adapter 统一处理
        text = content_div.get_text(separator="\n").strip() if content_div else ""
        
        title_tag = soup.find("h2") or soup.find("title")
        title = title_tag.get_text().strip() if title_tag else ""

        image_urls = []
        for img in soup.find_all("img"):
            src = img.get("bigimgsrc") or img.get("src")
            if src and "nosdn.127.net" in src:
                image_urls.append(src.split("?")[0])

        if not text and not image_urls:
            raise ParseException("无法解析 LOFTER 内容")

        return self.result(
            title=title or f"LOFTER 笔记 - {username}",
            text=text, 
            author=self.create_author(name=username),
            contents=self.create_image_contents(image_urls[:9]),
            url=url,
        )
