from __future__ import annotations

import time
from dataclasses import dataclass
from http import cookiejar
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlparse

from astrbot.api import logger


@dataclass(slots=True)
class Cookie:
    domain: str
    path: str
    name: str
    value: str
    secure: bool
    expires: int

    def is_expired(self) -> bool:
        return self.expires != 0 and self.expires < int(time.time())

    def match(self, domain: str, path: str, secure: bool) -> bool:
        if self.is_expired():
            return False
        if self.secure and not secure:
            return False
        if self.domain.startswith("."):
            if not domain.endswith(self.domain[1:]):
                return False
        elif domain != self.domain:
            return False
        return path.startswith(self.path)


class CookieJar:
    def __init__(self, cookie_dir: Path, name: str, domain: str, raw_cookies: str = ""):
        self.domain = domain
        self.cookie_file = cookie_dir / f"{name}_cookies.txt"
        self.cookies: list[Cookie] = []
        self.cookies_str = ""

        if raw_cookies.strip():
            self._load_from_cookies_str(raw_cookies)
            self.save_to_file()

        if self.cookie_file.exists():
            self.load_from_file()

    @staticmethod
    def clean_cookies_str(cookies_str: str) -> str:
        return cookies_str.replace("\n", "").replace("\r", "").strip()

    def get_cookie_header(self, path: str = "/", secure: bool = True) -> str:
        cookies = [
            cookie
            for cookie in self.cookies
            if cookie.match(self.domain, path, secure)
        ]
        return "; ".join(f"{cookie.name}={cookie.value}" for cookie in cookies)

    def get_cookie_header_for_url(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.hostname:
            return ""
        return self.get_cookie_header(
            path=parsed.path or "/",
            secure=parsed.scheme == "https",
        )

    def _sync_cookies_str(self) -> None:
        self.cookies_str = "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.cookies)

    def _load_from_cookies_str(self, cookies_str: str) -> None:
        cleaned = self.clean_cookies_str(cookies_str)
        if not cleaned:
            return
        self.cookies.clear()
        for item in cleaned.split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            name, value = item.split("=", 1)
            if not name.strip():
                continue
            self.cookies.append(
                Cookie(
                    domain=f".{self.domain}",
                    path="/",
                    name=name.strip(),
                    value=value.strip(),
                    secure=True,
                    expires=0,
                )
            )
        self._sync_cookies_str()

    def save_to_file(self) -> None:
        jar = cookiejar.MozillaCookieJar(str(self.cookie_file))
        for cookie in self.cookies:
            jar.set_cookie(
                cookiejar.Cookie(
                    version=0,
                    name=cookie.name,
                    value=cookie.value,
                    port=None,
                    port_specified=False,
                    domain=cookie.domain,
                    domain_specified=True,
                    domain_initial_dot=cookie.domain.startswith("."),
                    path=cookie.path,
                    path_specified=True,
                    secure=cookie.secure,
                    expires=cookie.expires,
                    discard=cookie.expires == 0,
                    comment=None,
                    comment_url=None,
                    rest={"HttpOnly": ""},
                    rfc2109=False,
                )
            )
        jar.save(ignore_discard=True, ignore_expires=True)

    def load_from_file(self) -> None:
        jar = cookiejar.MozillaCookieJar(str(self.cookie_file))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
        except Exception:
            logger.warning(f"[link_parser] failed to load cookies: {self.cookie_file}")
            return
        self.cookies = [
            Cookie(
                domain=item.domain,
                path=item.path,
                name=item.name,
                value=item.value or "",
                secure=item.secure,
                expires=item.expires or 0,
            )
            for item in jar
        ]
        self._sync_cookies_str()

    def update_from_response(self, set_cookie_headers: list[str]) -> None:
        changed = False
        for header in set_cookie_headers:
            parsed = SimpleCookie()
            parsed.load(header)
            for name, morsel in parsed.items():
                value = morsel.value
                path = morsel["path"] or "/"
                domain = morsel["domain"] or f".{self.domain}"
                secure = bool(morsel["secure"])
                expires = 0
                if morsel["expires"]:
                    try:
                        expires = int(
                            time.mktime(
                                time.strptime(
                                    morsel["expires"], "%a, %d-%b-%Y %H:%M:%S %Z"
                                )
                            )
                        )
                    except Exception:
                        expires = 0

                existing = next(
                    (
                        item
                        for item in self.cookies
                        if item.name == name and item.domain == domain and item.path == path
                    ),
                    None,
                )
                if existing:
                    existing.value = value
                    existing.secure = secure
                    existing.expires = expires
                else:
                    self.cookies.append(
                        Cookie(
                            domain=domain,
                            path=path,
                            name=name,
                            value=value,
                            secure=secure,
                            expires=expires,
                        )
                    )
                changed = True

        if changed:
            self.cookies = [cookie for cookie in self.cookies if not cookie.is_expired()]
            self._sync_cookies_str()
            self.save_to_file()
