from __future__ import annotations

import argparse
from pathlib import Path

from ..modnet.runtime import ModNetRuntimeConfig, run_modnet_loopback, run_modnet_srt
from ..registry import get_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BEN2 TensorRT live runtime")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--fps", type=int, default=30, help="Output fps")
    parser.add_argument("--v4l2-device", default="/dev/video10", help="v4l2loopback device path")
    parser.add_argument("--srt-url", default=None, help="Optional SRT URL")
    parser.add_argument("--mode", choices=["loopback", "srt"], default="loopback", help="Output mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = get_model_spec("ben2-trt")
    engine_path = Path(spec.metadata["engine_path"])
    config = ModNetRuntimeConfig(
        engine_path=engine_path,
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
