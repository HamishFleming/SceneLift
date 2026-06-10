from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...logging_controller import get_logger
from ..tensorrt.session import TensorRTSession, load_engine
from .trt_backend import ModNetTensorRTRemover


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModNetRuntimeConfig:
    engine_path: Path
    webcam_index: int = 0
    width: int = 512
    height: int = 512
    fps: int = 30
    v4l2_device: str = "/dev/video10"
    srt_url: str | None = None
    use_ffmpeg: bool = False


def _parse_shape(text: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must have two comma-separated integers")
    return tuple(parts)  # type: ignore[return-value]


def _write_loop(session: TensorRTSession, config: ModNetRuntimeConfig, out_sink) -> None:
    cap = cv2.VideoCapture(config.webcam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    remover = ModNetTensorRTRemover(
        engine_path=config.engine_path,
        input_size=(config.width, config.height),
        session=session,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgba = remover.remove(frame)
            out_sink(rgba)
    finally:
        cap.release()


def run_modnet_loopback(config: ModNetRuntimeConfig) -> None:
    session = load_engine(config.engine_path)
    out_cam = cv2.VideoWriter(
        config.v4l2_device,
        cv2.CAP_V4L2,
        cv2.VideoWriter_fourcc(*"YUYV"),
        config.fps,
        (config.width, config.height),
        True,
    )
    if not out_cam.isOpened():
        raise RuntimeError(f"Cannot open {config.v4l2_device} (v4l2loopback)")

    logger.info("Running MODNet loopback to %s", config.v4l2_device)
    try:
        _write_loop(session, config, lambda rgba: out_cam.write(cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)))
    finally:
        out_cam.release()
        session.close()


def run_modnet_srt(config: ModNetRuntimeConfig) -> None:
    if not config.srt_url:
        raise ValueError("srt_url is required for SRT output")

    session = load_engine(config.engine_path)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{config.width}x{config.height}",
            "-r",
            str(config.fps),
            "-i",
            "-",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-b:v",
            "6M",
            "-g",
            str(config.fps),
            "-f",
            "mpegts",
            config.srt_url,
        ],
        stdin=subprocess.PIPE,
    )
    if ffmpeg.stdin is None:
        raise RuntimeError("Could not open ffmpeg stdin")

    logger.info("Running MODNet SRT to %s", config.srt_url)

    try:
        _write_loop(
            session,
            config,
            lambda rgba: ffmpeg.stdin.write(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGB).tobytes()),
        )
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MODNet TensorRT live runtime")
    parser.add_argument("--engine-path", required=True, help="TensorRT engine file")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=512, help="Runtime width")
    parser.add_argument("--height", type=int, default=512, help="Runtime height")
    parser.add_argument("--fps", type=int, default=30, help="Output fps")
    parser.add_argument("--v4l2-device", default="/dev/video10", help="v4l2loopback device path")
    parser.add_argument("--srt-url", default=None, help="Optional SRT URL")
    parser.add_argument("--mode", choices=["loopback", "srt"], default="loopback", help="Output mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModNetRuntimeConfig(
        engine_path=Path(args.engine_path),
        webcam_index=args.webcam_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        v4l2_device=args.v4l2_device,
        srt_url=args.srt_url,
        use_ffmpeg=args.mode == "srt",
    )
    if args.mode == "loopback":
        run_modnet_loopback(config)
    else:
        run_modnet_srt(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
