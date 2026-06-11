from __future__ import annotations

import argparse

from ..io import is_image_output, parse_source
from ..logging_controller import get_logger, setup_logging
from .pipeline import RunConfig, run_image_directory, run_image_file, run_video_or_camera
from .registry import create_upscaler, list_upscaler_specs


logger = get_logger(__name__)


def available_upscale_method_choices() -> list[str]:
    return sorted({spec.key for spec in list_upscaler_specs()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale images, folders, or camera/video streams with Real-ESRGAN")
    parser.add_argument("--input", "-i", required=True, help="Input file path or camera spec like camera:0")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path or virtualcam for loopback camera output",
    )
    parser.add_argument(
        "--method",
        default="realesrgan-x4plus",
        choices=available_upscale_method_choices(),
        help="Upscaling model to use",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local Real-ESRGAN checkpoint path",
    )
    parser.add_argument("--tile", type=int, default=0, help="Tile size for tiled upscaling")
    parser.add_argument("--tile-pad", type=int, default=10, help="Padding around each tile")
    parser.add_argument("--pre-pad", type=int, default=0, help="Padding applied before upscaling")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device such as cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Force half precision when using CUDA",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable half precision",
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
        "--virtualcam-no-sleep",
        action="store_true",
        help="Do not pace virtualcam output to its nominal FPS; run as fast as inference allows",
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
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    input_source = parse_source(args.input)
    if args.half and args.no_half:
        raise SystemExit("--half and --no-half cannot be used together")
    half = None
    if args.half:
        half = True
    elif args.no_half:
        half = False

    upscaler = create_upscaler(
        args.method,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        device=args.device,
        half=half,
    )

    if input_source.kind == "image":
        if args.output == "virtualcam":
            raise SystemExit("virtualcam output is only supported for video or camera input")
        if not is_image_output(args.output):
            raise SystemExit("image input requires an image output path such as .png or .jpg")
        run_image_file(str(input_source.value), args.output, upscaler)
        return 0

    if input_source.kind == "directory":
        if args.output == "virtualcam":
            raise SystemExit("virtualcam output is only supported for video or camera input")
        if is_image_output(args.output):
            raise SystemExit("directory input requires an output directory, not a single image file")
        run_image_directory(str(input_source.value), args.output, upscaler)
        return 0

    if input_source.kind not in {"video", "camera"}:
        raise SystemExit(f"Unsupported input type: {input_source.kind}")
    if args.output != "virtualcam" and is_image_output(args.output):
        raise SystemExit("video or camera input requires a video file output or virtualcam")

    run_video_or_camera(
        RunConfig(
            input_source=input_source,
            output=args.output,
            upscaler=upscaler,
            max_frames=args.max_frames,
            virtualcam_device=args.virtualcam_device,
            virtualcam_sleep=not args.virtualcam_no_sleep,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
