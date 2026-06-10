from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .io import open_capture, parse_source
from .logging_controller import get_logger, setup_logging


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video frames to a folder")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to write extracted frames into")
    parser.add_argument(
        "--format",
        choices=["png", "webp"],
        default="png",
        help="Output image format for extracted frames",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Filename prefix for extracted frames",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting frame index used in output filenames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to extract",
    )
    parser.add_argument(
        "--webp-quality",
        type=int,
        default=90,
        help="WebP quality value used when --format webp is selected",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write logs to in addition to stderr",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Emit JSON log records instead of human-readable log lines",
    )
    return parser


def _encode_params(output_format: str, webp_quality: int) -> list[int]:
    if output_format == "webp":
        return [cv2.IMWRITE_WEBP_QUALITY, max(1, min(100, webp_quality))]
    return []


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    source = parse_source(args.input)
    if source.kind not in {"video", "file"}:
        raise SystemExit("frame extraction requires a video file input")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = open_capture(source)
    extracted = 0
    params = _encode_params(args.format, args.webp_quality)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = args.start_index + extracted
            frame_name = f"{args.prefix}_{frame_index:06d}.{args.format}"
            frame_path = output_dir / frame_name
            if not cv2.imwrite(str(frame_path), frame, params):
                raise RuntimeError(f"Could not write frame to {frame_path}")

            extracted += 1
            if extracted % 100 == 0:
                logger.info("Extracted %d frames", extracted)
            if args.max_frames is not None and extracted >= args.max_frames:
                logger.info("Reached max_frames=%d", args.max_frames)
                break
    finally:
        capture.release()
        logger.info("Frame extraction finished frames=%d output_dir=%s", extracted, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
