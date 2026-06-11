from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..io import SourceSpec, is_image_output, make_video_writer, open_capture
from ..logging_controller import get_logger
from .base import ImageUpscaler


logger = get_logger(__name__)


@dataclass
class RunConfig:
    input_source: SourceSpec
    output: str
    upscaler: ImageUpscaler
    max_frames: int | None = None
    virtualcam_device: str | None = None
    virtualcam_sleep: bool = True


def _resize_bgr(frame_bgr, target_shape: tuple[int, int]):
    target_height, target_width = target_shape
    if frame_bgr.shape[:2] == (target_height, target_width):
        return frame_bgr
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Resizing backend output from %s to %s", frame_bgr.shape[:2], target_shape)
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _put_latest(frame_queue: queue.Queue, item) -> None:
    while True:
        try:
            frame_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                return


def _capture_worker(
    capture: cv2.VideoCapture,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    max_frames: int | None,
) -> None:
    captured = 0
    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                logger.info("Capture ended after %d frames", captured)
                break

            _put_latest(frame_queue, frame)
            captured += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Captured frame %d shape=%s", captured, getattr(frame, "shape", None))

            if max_frames is not None and captured >= max_frames:
                logger.info("Reached capture max_frames=%d", max_frames)
                break
    finally:
        _put_latest(frame_queue, None)
        logger.info("Capture worker finished")


def save_image(output_path: str, frame_bgr) -> None:
    path = Path(output_path)
    logger.info("Saving image to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Could not write image to {path}")


def run_image_file(input_path: str, output_path: str, upscaler: ImageUpscaler) -> None:
    logger.info("Processing image input=%s output=%s method=%s", input_path, output_path, upscaler.name)
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")
    try:
        upscaled = upscaler.upscale(image)
        save_image(output_path, upscaled)
    finally:
        close = getattr(upscaler, "close", None)
        if callable(close):
            close()


def run_image_directory(input_dir: str, output_dir: str, upscaler: ImageUpscaler) -> None:
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    logger.info("Processing image directory input=%s output=%s method=%s", source_dir, target_dir, upscaler.name)
    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise RuntimeError(f"No supported image files were found in {source_dir}")

    try:
        processed = 0
        for path in image_paths:
            logger.debug("Directory frame %d input=%s", processed + 1, path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("Skipping unreadable image %s", path)
                continue
            upscaled = upscaler.upscale(image)
            stem = path.stem
            output_path = target_dir / f"{stem}.png"
            save_image(str(output_path), upscaled)
            processed += 1
            if processed % 30 == 0:
                logger.info("Processed %d directory frames", processed)
        logger.info("Directory processing finished frames=%d output_dir=%s", processed, target_dir)
    finally:
        close = getattr(upscaler, "close", None)
        if callable(close):
            close()


def run_video_or_camera(config: RunConfig) -> None:
    logger.info(
        "Processing stream input_kind=%s output=%s method=%s",
        config.input_source.kind,
        config.output,
        config.upscaler.name,
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
        scale = int(getattr(config.upscaler, "scale", 1) or 1)
        output_width = width * max(1, scale)
        output_height = height * max(1, scale)
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
                width=output_width,
                height=output_height,
                fps=fps,
                device=config.virtualcam_device,
            )
            logger.info("Opened virtual camera device=%s", config.virtualcam_device)
        else:
            Path(config.output).parent.mkdir(parents=True, exist_ok=True)
            writer = make_video_writer(config.output, fps, output_width, output_height)

        frame_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        capture_thread = threading.Thread(
            target=_capture_worker,
            args=(capture, frame_queue, stop_event, config.max_frames),
            name="bgremoval-upscale-capture",
            daemon=True,
        )
        capture_thread.start()

        frame_count = 0
        started = time.perf_counter()
        while True:
            frame = frame_queue.get()
            if frame is None:
                break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Frame %d input_shape=%s", frame_count + 1, getattr(frame, "shape", None))
            upscaled = config.upscaler.upscale(frame)
            upscaled = _resize_bgr(upscaled, (output_height, output_width))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Frame %d output_shape=%s", frame_count + 1, getattr(upscaled, "shape", None))

            if virtualcam is not None:
                rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
                virtualcam.send(rgb)
                if config.virtualcam_sleep:
                    virtualcam.sleep_until_next_frame()
            else:
                writer.write(upscaled)

            frame_count += 1
            elapsed = time.perf_counter() - started
            if frame_count % 30 == 0 and elapsed > 0:
                logger.info("Processed %d frames effective_fps=%.2f", frame_count, frame_count / elapsed)
    finally:
        try:
            stop_event.set()
        except UnboundLocalError:
            pass
        try:
            capture_thread.join(timeout=2.0)
        except UnboundLocalError:
            pass
        capture.release()
        if "writer" in locals() and writer is not None:
            writer.release()
        if "virtualcam" in locals() and virtualcam is not None:
            virtualcam.close()
        close = getattr(config.upscaler, "close", None)
        if callable(close):
            close()
        logger.info("Stream processing finished")
