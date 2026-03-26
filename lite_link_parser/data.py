from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Platform:
    name: str
    display_name: str


@dataclass(slots=True)
class Author:
    name: str
    avatar_url: str | None = None
    description: str | None = None


@dataclass(slots=True)
class MediaContent:
    kind: str
    url: str
    cover_url: str | None = None
    text: str | None = None
    alt: str | None = None
    duration: float = 0.0
    name: str | None = None


@dataclass(slots=True)
class ParseResult:
    platform: Platform
    author: Author | None = None
    title: str | None = None
    text: str | None = None
    timestamp: int | None = None
    url: str | None = None
    contents: list[MediaContent] = field(default_factory=list)
    extra_info: str | None = None
    repost: "ParseResult | None" = None

    @property
    def image_contents(self) -> list[MediaContent]:
        return [item for item in self.contents if item.kind in {"image", "graphics"}]

    @property
    def video_contents(self) -> list[MediaContent]:
        return [item for item in self.contents if item.kind == "video"]

    @property
    def audio_contents(self) -> list[MediaContent]:
        return [item for item in self.contents if item.kind == "audio"]
