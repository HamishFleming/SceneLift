from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .logging_controller import get_logger, setup_logging


logger = get_logger(__name__)
_DEFAULT_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def _collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    candidates: list[Path] = []
    iterator = input_dir.rglob if recursive else input_dir.glob
    for pattern in _DEFAULT_GLOBS:
        candidates.extend(sorted(iterator(pattern)))
    return sorted({path.resolve() for path in candidates if path.is_file()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a calibration image set from a directory of frames")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing source frames or sample images")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to write the calibration set into")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Maximum number of images to copy into the calibration set",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images recursively under the input directory",
    )
    parser.add_argument(
        "--prefix",
        default="calib",
        help="Filename prefix used for the copied calibration images",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index used in the output filenames",
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

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(input_dir, args.recursive)
    if not images:
        raise SystemExit(f"No calibration images found under {input_dir}")

    selected = images[: args.max_samples] if args.max_samples > 0 else images
    if not selected:
        raise SystemExit("No calibration images were selected")

    copied = 0
    for index, source in enumerate(selected, start=args.start_index):
        destination = output_dir / f"{args.prefix}_{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied += 1
        if copied % 100 == 0:
            logger.info("Prepared %d calibration images", copied)

    logger.info("Calibration set ready copied=%d output_dir=%s", copied, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
