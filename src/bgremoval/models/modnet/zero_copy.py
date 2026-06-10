from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import ModNetRuntimeConfig, run_modnet_srt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MODNet TensorRT zero-copy-style SRT pipeline")
    parser.add_argument("--engine-path", required=True, help="TensorRT engine file")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--fps", type=int, default=30, help="Output fps")
    parser.add_argument("--srt-url", required=True, help="SRT destination URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_modnet_srt(
        ModNetRuntimeConfig(
            engine_path=Path(args.engine_path),
            webcam_index=args.webcam_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            srt_url=args.srt_url,
            use_ffmpeg=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
