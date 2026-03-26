from __future__ import annotations

from abc import ABC
from asyncio import TimeoutError, sleep
from collections.abc import Callable, Coroutine
from pathlib import Path
from re import Match, Pattern, compile
from typing import Any, ClassVar, TypeVar, cast

from aiohttp import ClientError, ClientSession, ClientTimeout

from .constants import ANDROID_HEADER, COMMON_HEADER, IOS_HEADER
from .data import Author, MediaContent, ParseResult
from .exception import ParseException, RedirectException

T = TypeVar("T", bound="BaseLiteParser")
HandlerFunc = Callable[[T, Match[str]], Coroutine[Any, Any, ParseResult]]
KeyPatterns = list[tuple[str, Pattern[str]]]
_KEY_PATTERNS = "_key_patterns"


def handle(keyword: str, pattern: str):
    def decorator(func: HandlerFunc[T]) -> HandlerFunc[T]:
        if not hasattr(func, _KEY_PATTERNS):
            setattr(func, _KEY_PATTERNS, [])
        key_patterns: KeyPatterns = getattr(func, _KEY_PATTERNS)
        key_patterns.append((keyword, compile(pattern)))
        return func

    return decorator


class BaseLiteParser:
    _registry: ClassVar[list[type["BaseLiteParser"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if ABC not in cls.__bases__:
            BaseLiteParser._registry.append(cls)

        cls._handlers = {}
        cls._key_patterns = []

        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and hasattr(attr, _KEY_PATTERNS):
                key_patterns: KeyPatterns = getattr(attr, _KEY_PATTERNS)
                handler = cast(HandlerFunc, attr)
                for keyword, pattern in key_patterns:
                    cls._handlers[keyword] = handler
                    cls._key_patterns.append((keyword, pattern))

        cls._key_patterns.sort(key=lambda item: -len(item[0]))

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.headers = COMMON_HEADER.copy()
        self.ios_headers = IOS_HEADER.copy()
        self.android_headers = ANDROID_HEADER.copy()
        self._session: ClientSession | None = None

    @property
    def site_config(self) -> dict[str, Any]:
        return self.config["sites"].get(self.platform.name, {})

    @property
    def proxy(self) -> str | None:
        site_proxy = self.site_config.get("proxy")
        return site_proxy or self.config.get("proxy") or None

    @property
    def session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                timeout=ClientTimeout(total=float(self.config.get("timeout", 12.0))),
            )
        return self._session

    async def close_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def parse(self, keyword: str, searched: Match[str]) -> ParseResult:
        return await self._handlers[keyword](self, searched)

    @classmethod
    def search_url(cls, url: str) -> tuple[str, Match[str]]:
        for keyword, pattern in cls._key_patterns:
            if keyword not in url:
                continue
            matched = pattern.search(url)
            if matched:
                return keyword, matched
        raise ParseException(f"no parser matched: {url}")

    async def parse_with_redirect(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> ParseResult:
        redirect_url = await self.get_redirect_url(url, headers=headers or self.headers)
        if redirect_url == url:
            raise ParseException(f"redirect failed: {url}")
        keyword, searched = self.search_url(redirect_url)
        return await self.parse(keyword, searched)

    async def get_redirect_url(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        request_headers = headers or COMMON_HEADER.copy()
        retries = 2
        for attempt in range(retries + 1):
            try:
                async with self.session.get(
                    url,
                    headers=request_headers,
                    allow_redirects=False,
                    proxy=self.proxy,
                    ssl=False,
                ) as response:
                    if response.status >= 400:
                        raise ClientError(f"redirect check {response.status} {response.reason}")
                    return response.headers.get("Location", url)
            except (ClientError, TimeoutError):
                if attempt < retries:
                    await sleep(1 + attempt)
                    continue
                raise RedirectException()
        raise RedirectException()

    def result(self, **kwargs: Any) -> ParseResult:
        return ParseResult(platform=self.platform, **kwargs)

    @staticmethod
    def create_author(
        name: str | None,
        avatar_url: str | None = None,
        description: str | None = None,
    ) -> Author | None:
        clean_name = (name or "").strip()
        if not clean_name:
            return None
        return Author(name=clean_name, avatar_url=avatar_url, description=description)

    @staticmethod
    def create_image_content(url: str, text: str | None = None, alt: str | None = None) -> MediaContent:
        return MediaContent(kind="image", url=url, text=text, alt=alt)

    @classmethod
    def create_image_contents(cls, image_urls: list[str]) -> list[MediaContent]:
        return [cls.create_image_content(url) for url in image_urls if url]

    @staticmethod
    def create_video_content(
        url: str,
        cover_url: str | None = None,
        duration: float = 0.0,
    ) -> MediaContent:
        return MediaContent(kind="video", url=url, cover_url=cover_url, duration=duration)

    @staticmethod
    def create_audio_content(
        url: str,
        cover_url: str | None = None,
        duration: float = 0.0,
        name: str | None = None,
    ) -> MediaContent:
        return MediaContent(
            kind="audio",
            url=url,
            cover_url=cover_url,
            duration=duration,
            name=name,
        )

    @staticmethod
    def create_graphics_content(url: str, text: str | None = None, alt: str | None = None) -> MediaContent:
        return MediaContent(kind="graphics", url=url, text=text, alt=alt)

    @staticmethod
    def ensure_cookie_dir(base_dir: str | Path) -> Path:
        cookie_dir = Path(base_dir)
        cookie_dir.mkdir(parents=True, exist_ok=True)
        return cookie_dir
