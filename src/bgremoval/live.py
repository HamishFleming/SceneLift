from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .io import SourceSpec, open_capture
from .logging_controller import get_logger
from .methods import BackgroundRemover
from .pipeline import composite_frame


logger = get_logger(__name__)


@dataclass(frozen=True)
class LiveConfig:
    input_source: SourceSpec
    method: BackgroundRemover
    background_color: tuple[int, int, int] = (0, 0, 0)
    virtualcam_device: str | None = None
    max_frames: int | None = None
    max_dimension: int | None = 1280
    target_fps: float | None = None


def _compute_resize_size(width: int, height: int, max_dimension: int | None) -> tuple[int, int]:
    if not max_dimension or max_dimension <= 0:
        return width, height
    largest = max(width, height)
    if largest <= max_dimension:
        return width, height
    scale = max_dimension / float(largest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _put_latest(frame_queue: queue.Queue[np.ndarray | None], item: np.ndarray | None) -> None:
    while True:
        try:
            frame_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass


def _capture_worker(
    capture: cv2.VideoCapture,
    frame_queue: queue.Queue[np.ndarray | None],
    stop_event: threading.Event,
    resize_to: tuple[int, int] | None,
    max_frames: int | None,
) -> None:
    captured = 0
    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                logger.info("Capture ended after %d frames", captured)
                break

            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

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


def run_live_virtualcam(config: LiveConfig) -> None:
    logger.info(
        "Starting live pipeline input_kind=%s method=%s output=virtualcam",
        config.input_source.kind,
        config.method.name,
    )
    capture = open_capture(config.input_source)
    try:
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            logger.debug("Capture backend does not support CAP_PROP_BUFFERSIZE")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1:
            fps = config.target_fps or 30.0
        elif config.target_fps:
            fps = min(float(fps), float(config.target_fps))

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read initial frame from source")
            height, width = frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out_width, out_height = _compute_resize_size(width, height, config.max_dimension)
        resize_to = (out_width, out_height) if (out_width, out_height) != (width, height) else None
        logger.info(
            "Live stream source=%dx%d output=%dx%d fps=%.2f resize=%s",
            width,
            height,
            out_width,
            out_height,
            fps,
            resize_to is not None,
        )

        try:
            import pyvirtualcam
        except ImportError as exc:
            raise RuntimeError(
                "live virtualcam output requires pyvirtualcam. Install it with `pip install -e '.[virtualcam]'`."
            ) from exc

        virtualcam = pyvirtualcam.Camera(
            width=out_width,
            height=out_height,
            fps=fps,
            device=config.virtualcam_device,
        )
        logger.info("Opened virtual camera device=%s", config.virtualcam_device)

        frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        capture_thread = threading.Thread(
            target=_capture_worker,
            args=(capture, frame_queue, stop_event, resize_to, config.max_frames),
            name="bgremoval-capture",
            daemon=True,
        )
        capture_thread.start()

        processed = 0
        started = time.perf_counter()
        try:
            while True:
                frame = frame_queue.get()
                if frame is None:
                    break

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Live frame %d input_shape=%s", processed + 1, getattr(frame, "shape", None))

                rgba = config.method.remove(frame)
                bgr = composite_frame(rgba, config.background_color)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                virtualcam.send(rgb)
                virtualcam.sleep_until_next_frame()

                processed += 1
                elapsed = time.perf_counter() - started
                if processed % 30 == 0 and elapsed > 0:
                    logger.info("Live processed=%d effective_fps=%.2f", processed, processed / elapsed)
        finally:
            stop_event.set()
            capture_thread.join(timeout=2.0)
            virtualcam.close()
            elapsed = time.perf_counter() - started
            if elapsed > 0:
                logger.info("Live stream finished processed=%d average_fps=%.2f", processed, processed / elapsed)
    finally:
        capture.release()
