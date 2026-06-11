from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .logging_controller import get_logger


logger = get_logger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    value: str | int


def parse_source(raw: str) -> SourceSpec:
    token = raw.strip()
    logger.debug("Parsing input source %s", token)
    if token.startswith("camera:"):
        return SourceSpec(kind="camera", value=int(token.split(":", 1)[1]))
    path = Path(token)
    if path.exists():
        if path.is_dir():
            return SourceSpec(kind="directory", value=str(path))
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return SourceSpec(kind="image", value=str(path))
        if suffix in VIDEO_EXTENSIONS:
            return SourceSpec(kind="video", value=str(path))
        return SourceSpec(kind="file", value=str(path))
    if token.isdigit():
        return SourceSpec(kind="camera", value=int(token))
    raise FileNotFoundError(f"Input source not found: {raw}")


def is_image_output(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_output(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def open_capture(source: SourceSpec) -> cv2.VideoCapture:
    if source.kind == "camera":
        logger.info("Opening camera %s", source.value)
        capture = cv2.VideoCapture(int(source.value))
        # Keep the live buffer shallow so slow inference does not build a long backlog.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif source.kind in {"video", "file"}:
        logger.info("Opening video/file source %s", source.value)
        capture = cv2.VideoCapture(str(source.value))
    else:
        raise ValueError(f"open_capture does not support source kind {source.kind}")
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input source {source.value}")
    return capture


def make_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    suffix = Path(path).suffix.lower()
    if suffix in {".avi"}:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif suffix in {".mov"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    logger.info("Opening video writer path=%s fps=%.2f size=%dx%d", path, fps, width, height)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer
