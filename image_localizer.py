import hashlib
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from astrbot.api import logger


class ImageLocalizer:
    def __init__(
        self,
        *,
        plugin_dir: Path,
        enabled: bool,
        directory: str,
        timeout: float,
        max_bytes: int,
        cleanup_max_age_hours: float,
        convert_gif_to_jpg: bool,
    ):
        self.enabled = enabled
        self.directory = self._resolve_path(plugin_dir, directory)
        self.timeout = max(float(timeout), 1.0)
        self.max_bytes = int(max_bytes)
        self.cleanup_max_age_hours = float(cleanup_max_age_hours)
        self.convert_gif_to_jpg = convert_gif_to_jpg

    @staticmethod
    def from_config(config: dict, plugin_dir: Path) -> "ImageLocalizer":
        return ImageLocalizer(
            plugin_dir=plugin_dir,
            enabled=bool(config.get("enable_image_localization", False)),
            directory=config.get("image_localization_dir", "data/localized_images"),
            timeout=float(config.get("image_localization_timeout", 10.0)),
            max_bytes=int(config.get("image_localization_max_bytes", 20 * 1024 * 1024)),
            cleanup_max_age_hours=float(
                config.get("image_localization_cleanup_max_age_hours", 24.0)
            ),
            convert_gif_to_jpg=bool(
                config.get("image_localization_convert_gif_to_jpg", True)
            ),
        )

    @staticmethod
    def _resolve_path(plugin_dir: Path, path_value: str) -> Path:
        path = Path(str(path_value or "data/localized_images")).expanduser()
        if not path.is_absolute():
            path = plugin_dir / path
        return path

    @staticmethod
    def _is_http_url(value: str) -> bool:
        try:
            return urlparse(value or "").scheme in {"http", "https"}
        except Exception:
            return False

    @staticmethod
    def _extension_from_response(url: str, content_type: str) -> str:
        content_type = (content_type or "").split(";", 1)[0].strip().lower()
        guessed = mimetypes.guess_extension(content_type) if content_type else ""
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed

        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
            return suffix
        return ".jpg"

    def cleanup(self) -> None:
        if self.cleanup_max_age_hours <= 0 or not self.directory.exists():
            return

        cutoff = time.time() - self.cleanup_max_age_hours * 3600
        removed_count = 0
        for path in self.directory.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed_count += 1
            except Exception as exc:
                logger.debug(f"[消息防抖动] 清理本地化图片失败: {path} | {exc}")

        if removed_count:
            logger.info(f"[消息防抖动] 已清理过期本地化图片: {removed_count} 个")

    async def localize(self, image_refs: list[str]) -> list[str]:
        if not self.enabled or not image_refs:
            return image_refs

        self.cleanup()
        self.directory.mkdir(parents=True, exist_ok=True)

        localized = []
        success_count = 0
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for image_ref in image_refs:
                if not self._is_http_url(image_ref):
                    localized.append(image_ref)
                    continue
                try:
                    localized.append(await self._download(session, image_ref))
                    success_count += 1
                except Exception as exc:
                    logger.warning(f"[消息防抖动] 图片本地化失败，回退原始 URL: {exc}")
                    localized.append(image_ref)

        if success_count:
            logger.info(
                f"[消息防抖动] 图片本地化完成 | 成功: {success_count}/{len(image_refs)} | 目录: {self.directory}"
            )
        return localized

    async def _download(self, session: aiohttp.ClientSession, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()

        async with session.get(url) as response:
            response.raise_for_status()
            ext = self._extension_from_response(url, response.headers.get("Content-Type", ""))
            target = self.directory / f"{digest}{ext}"
            converted_target = self.directory / f"{digest}.jpg"
            if (
                ext == ".gif"
                and self.convert_gif_to_jpg
                and converted_target.exists()
                and converted_target.stat().st_size > 0
            ):
                return str(converted_target)
            if target.exists() and target.stat().st_size > 0:
                return self._maybe_convert_gif(target, digest)

            size = 0
            chunks = []
            async for chunk in response.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError(f"image too large: {size} bytes")
                chunks.append(chunk)

            target.write_bytes(b"".join(chunks))
            return self._maybe_convert_gif(target, digest)

    def _maybe_convert_gif(self, path: Path, digest: str) -> str:
        if path.suffix.lower() != ".gif" or not self.convert_gif_to_jpg:
            return str(path)
        return self._convert_gif_to_jpg(path, digest)

    def _convert_gif_to_jpg(self, gif_path: Path, digest: str) -> str:
        try:
            from PIL import Image
        except Exception as exc:
            logger.warning(f"[消息防抖动] Pillow 不可用，无法将 GIF 转为 JPG: {exc}")
            return str(gif_path)

        jpg_path = self.directory / f"{digest}.jpg"
        if jpg_path.exists() and jpg_path.stat().st_size > 0:
            return str(jpg_path)

        try:
            with Image.open(gif_path) as image:
                image.seek(0)
                frame = image.convert("RGBA")
                background = Image.new("RGB", frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.getchannel("A"))
                background.save(jpg_path, "JPEG", quality=95)
            try:
                gif_path.unlink()
            except Exception:
                pass
            logger.info(f"[消息防抖动] GIF 已转换为 JPG 第一帧: {jpg_path}")
            return str(jpg_path)
        except Exception as exc:
            logger.warning(f"[消息防抖动] GIF 转 JPG 失败，保留原 GIF: {exc}")
            return str(gif_path)
