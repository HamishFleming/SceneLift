from __future__ import annotations

import argparse

from .health import run_healthcheck
from ..logging_controller import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a startup healthcheck for a Real-ESRGAN upscaler")
    parser.add_argument("--input", "-i", required=True, help="Input file path or camera spec like camera:0")
    parser.add_argument("--method", default="realesrgan-x4plus", help="Upscaler to probe")
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)
    run_healthcheck(args.input, args.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
