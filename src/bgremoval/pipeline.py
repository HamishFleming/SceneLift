from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .logging_controller import get_logger
from .io import SourceSpec, make_video_writer, open_capture
from .methods import BackgroundRemover


logger = get_logger(__name__)


@dataclass(frozen=True)
class RunConfig:
    input_source: SourceSpec
    output: str
    method: BackgroundRemover
    background_color: tuple[int, int, int] = (0, 0, 0)
    max_frames: int | None = None
    virtualcam_device: str | None = None


def _split_bgr(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bgr = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return bgr, alpha


def _resize_rgba(rgba: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    if rgba.shape[:2] == (target_height, target_width):
        return rgba
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Resizing backend output from %s to %s", rgba.shape[:2], target_shape)
    return cv2.resize(rgba, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def composite_frame(rgba: np.ndarray, background_color: tuple[int, int, int]) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected RGBA frame")
    bgr, alpha = _split_bgr(rgba)
    bg = np.full_like(bgr, background_color, dtype=np.uint8)
    blended = (bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return blended


def _close_remover(remover: BackgroundRemover) -> None:
    close = getattr(remover, "close", None)
    if callable(close):
        close()


def save_image(output_path: str, rgba: np.ndarray, background_color: tuple[int, int, int]) -> None:
    path = Path(output_path)
    logger.info("Saving image to %s", path)
    if path.suffix.lower() == ".png":
        rgba_out = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
        Image.fromarray(rgba_out).save(path)
        return

    bgr = composite_frame(rgba, background_color)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)


def run_image_file(input_path: str, output_path: str, remover: BackgroundRemover, background_color: tuple[int, int, int]) -> None:
    logger.info("Processing image input=%s output=%s method=%s", input_path, output_path, remover.name)
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")
    try:
        rgba = remover.remove(image)
        rgba = _resize_rgba(rgba, image.shape[:2])
        save_image(output_path, rgba, background_color)
    finally:
        _close_remover(remover)


def run_video_or_camera(config: RunConfig) -> None:
    logger.info(
        "Processing stream input_kind=%s output=%s method=%s",
        config.input_source.kind,
        config.output,
        config.method.name,
    )
    capture = open_capture(config.input_source)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read initial frame from source")
            height, width = frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = None
        virtualcam = None

        if config.output == "virtualcam":
            try:
                import pyvirtualcam
            except ImportError as exc:
                raise RuntimeError(
                    "virtualcam output requires pyvirtualcam. Install it with `pip install -e '.[virtualcam]'`."
                ) from exc

            virtualcam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                device=config.virtualcam_device,
            )
            logger.info("Opened virtual camera device=%s", config.virtualcam_device)
        else:
            writer = make_video_writer(config.output, fps, width, height)

        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Frame %d input_shape=%s", frame_count + 1, getattr(frame, "shape", None))
            rgba = config.method.remove(frame)
            rgba = _resize_rgba(rgba, frame.shape[:2])
            bgr = composite_frame(rgba, config.background_color)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Frame %d output_shape=%s", frame_count + 1, getattr(bgr, "shape", None))

            if virtualcam is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                virtualcam.send(rgb)
                virtualcam.sleep_until_next_frame()
            else:
                writer.write(bgr)

            frame_count += 1
            if frame_count % 30 == 0:
                logger.info("Processed %d frames", frame_count)
            if config.max_frames is not None and frame_count >= config.max_frames:
                logger.info("Reached max_frames=%d", config.max_frames)
                break
    finally:
        capture.release()
        if "writer" in locals() and writer is not None:
            writer.release()
        if "virtualcam" in locals() and virtualcam is not None:
            virtualcam.close()
        _close_remover(config.method)
        logger.info("Stream processing finished")
