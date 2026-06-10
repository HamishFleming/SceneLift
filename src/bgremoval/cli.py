from __future__ import annotations

import argparse

from .logging_controller import get_logger, setup_logging
from .io import is_image_output, parse_source
from .methods import create_remover
from .live import LiveConfig, run_live_virtualcam
from .pipeline import RunConfig, run_image_file, run_video_or_camera


logger = get_logger(__name__)


def _parse_color(text: str) -> tuple[int, int, int]:
    value = text.strip()
    if value.startswith("#"):
        value = value[1:]
    if "," in value:
        parts = value.split(",")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("background color must have exactly 3 comma-separated values")
        rgb = tuple(int(part) for part in parts)
    else:
        if len(value) != 6:
            raise argparse.ArgumentTypeError("background color must be #RRGGBB or R,G,B")
        rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("background color channels must be 0..255")
    return rgb  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Background removal prototype")
    parser.add_argument("--input", "-i", required=True, help="Input file path or camera spec like camera:0")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path or virtualcam for loopback camera output",
    )
    parser.add_argument(
        "--method",
        default="grabcut",
        choices=["grabcut", "rembg", "birefnet", "u2net-human-seg", "mediapipe-selfie-segmentation"],
        help="Background removal method to use",
    )
    parser.add_argument(
        "--background-color",
        default="#000000",
        type=_parse_color,
        help="Background color for non-transparent outputs, either #RRGGBB or R,G,B",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for camera/video input",
    )
    parser.add_argument(
        "--virtualcam-device",
        default=None,
        help="Optional virtual camera device name passed to pyvirtualcam",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the low-latency live pipeline when streaming to virtualcam",
    )
    parser.add_argument(
        "--live-max-dimension",
        type=int,
        default=1280,
        help="Downscale webcam frames to fit within this size before inference",
    )
    parser.add_argument(
        "--live-target-fps",
        type=float,
        default=None,
        help="Optional cap for live output fps",
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    input_source = parse_source(args.input)
    remover = create_remover(args.method)

    if input_source.kind == "image":
        if args.output == "virtualcam":
            raise SystemExit("virtualcam output is only supported for video or camera input")
        if not is_image_output(args.output):
            raise SystemExit("image input requires an image output path such as .png or .jpg")
        run_image_file(
            input_path=str(input_source.value),
            output_path=args.output,
            remover=remover,
            background_color=args.background_color,
        )
        return 0

    if input_source.kind not in {"video", "camera"}:
        raise SystemExit(f"Unsupported input type: {input_source.kind}")
    if args.output != "virtualcam" and is_image_output(args.output):
        raise SystemExit("video or camera input requires a video file output or virtualcam")

    if args.live or (input_source.kind == "camera" and args.output == "virtualcam"):
        if args.output != "virtualcam":
            raise SystemExit("live mode currently requires virtualcam output")
        logger.info("Using live pipeline from CLI")
        run_live_virtualcam(
            LiveConfig(
                input_source=input_source,
                method=remover,
                background_color=args.background_color,
                virtualcam_device=args.virtualcam_device,
                max_frames=args.max_frames,
                max_dimension=args.live_max_dimension,
                target_fps=args.live_target_fps,
            )
        )
        return 0

    config = RunConfig(
        input_source=input_source,
        output=args.output,
        method=remover,
        background_color=args.background_color,
        max_frames=args.max_frames,
        virtualcam_device=args.virtualcam_device,
    )
    run_video_or_camera(config)
    return 0
