from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from astrbot.api import logger

from ..base import BaseLiteParser, handle
from ..cookie import CookieJar
from ..data import Platform
from ..exception import ParseException

class ZhihuLiteParser(BaseLiteParser):
    platform: ClassVar[Platform] = Platform(name="zhihu", display_name="知乎")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Referer": "https://www.zhihu.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Upgrade-Insecure-Requests": "1"
        })
        
        cookie_dir = self.ensure_cookie_dir(self.config.get("cookie_dir", "data/cookies"))
        cookie_file = cookie_dir / "zhihu_cookies.txt"
        if cookie_file.exists():
            try:
                with open(cookie_file, "r", encoding="utf-8") as f:
                    # 强力加载 Cookie
                    self.headers["Cookie"] = f.read().strip()
            except Exception:
                pass

    @handle("zhihu.com/question", r"(?:www\.)?zhihu\.com/question/\d+/answer/(?P<answer_id>\d+)")
    async def _parse_answer(self, searched: re.Match[str]):
        url = searched.group(0)
        if not url.startswith("http"):
            url = f"https://{url}"
            
        # 修正：curl_cffi 使用 proxy 而非 proxies
        proxy_url = self.proxy if self.proxy else None
        
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            resp = await session.get(url, headers=self.headers, proxy=proxy_url, timeout=15)
            html_text = resp.text

        data = self._extract_initial_data(html_text)
        if not data:
            return self._parse_html_fallback(html_text, url, resp.status_code)

        try:
            entities = data.get("initialState", {}).get("entities", {})
            answers = entities.get("answers", {})
            answer_id = searched.group("answer_id")
            answer = answers.get(answer_id) or next(iter(answers.values()), {})

            author_name = answer.get("author", {}).get("name", "知乎用户")
            text, image_urls = self._clean_zhihu_html(answer.get("content", ""))
            
            questions = entities.get("questions", {})
            question = next(iter(questions.values()), {})
            title = question.get("title", "知乎回答")

            return self.result(
                title=title,
                text=text,
                author=self.create_author(name=author_name),
                contents=self.create_image_contents(image_urls[:5]),
                url=url,
            )
        except Exception as e:
            return self._parse_html_fallback(html_text, url, resp.status_code)

    @handle("zhuanlan.zhihu.com", r"zhuanlan\.zhihu\.com/p/(?P<article_id>\d+)")
    async def _parse_article(self, searched: re.Match[str]):
        article_id = searched.group("article_id")
        url = f"https://zhuanlan.zhihu.com/p/{article_id}"
        proxy_url = self.proxy if self.proxy else None

        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            resp = await session.get(url, headers=self.headers, proxy=proxy_url, timeout=15)
            html_text = resp.text

        data = self._extract_initial_data(html_text)
        if not data:
            return self._parse_html_fallback(html_text, url, resp.status_code)

        try:
            entities = data.get("initialState", {}).get("entities", {})
            articles = entities.get("articles", {})
            article = articles.get(article_id) or next(iter(articles.values()), {})

            author_name = article.get("author", {}).get("name", "知乎专栏")
            title = article.get("title", "知乎文章")
            text, image_urls = self._clean_zhihu_html(article.get("content", ""))

            return self.result(
                title=title,
                text=text,
                author=self.create_author(name=author_name),
                contents=self.create_image_contents(image_urls[:5]),
                url=url,
            )
        except Exception:
            return self._parse_html_fallback(html_text, url, resp.status_code)

    def _extract_initial_data(self, html_text: str) -> dict | None:
        matched = re.search(r'<script id="js-initialData" type="text/json">(.*?)</script>', html_text, re.S)
        if matched:
            try: return json.loads(matched.group(1))
            except: pass
        return None

    def _clean_zhihu_html(self, content_html: str) -> tuple[str, list[str]]:
        if not content_html: return "", []
        soup = BeautifulSoup(content_html, "html.parser")
        image_urls = []
        for img in soup.find_all("img"):
            src = img.get("data-actualsrc") or img.get("src")
            if src and "zhimg.com" in src: image_urls.append(src)
        return soup.get_text(separator="\n").strip(), image_urls

    def _parse_html_fallback(self, html_text: str, url: str, status_code: int):
        soup = BeautifulSoup(html_text, "html.parser")
        title_tag = soup.find("h1") or soup.find("title")
        
        # 诊断信息
        if "安全验证" in html_text or "验证码" in html_text or status_code == 403:
            title = "知乎反爬拦截"
            text = f"（状态码 {status_code}：触发了知乎滑块验证或 IP 封禁，必须更新 Cookie 或更换代理 IP）"
        elif title_tag:
            title = title_tag.get_text().strip()
            text = "（知乎返回了空正文，通常是由于未登录导致的内容屏蔽）"
        else:
            title = "知乎连接受阻"
            text = f"（无法解析页面。状态码: {status_code}。请确保已在 data/cookies/ 中放置了 zhihu_cookies.txt）"
        
        return self.result(title=title, text=text, author=self.create_author(name="知乎用户"), url=url)