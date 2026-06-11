from __future__ import annotations

import argparse
from dataclasses import asdict

from ..logging_controller import get_logger, setup_logging
from .health import benchmark_upscaler, load_reference_frame, log_environment_health
from .registry import list_upscaler_specs


logger = get_logger(__name__)


def available_upscale_method_choices() -> list[str]:
    return sorted({spec.key for spec in list_upscaler_specs()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Real-ESRGAN upscalers on the same input frame")
    parser.add_argument("--input", "-i", required=True, help="Input file path or camera spec like camera:0")
    parser.add_argument(
        "--methods",
        default="realesrgan-x4plus,realesrgan-x4plus-anime,realesrgan-x2plus",
        help="Comma-separated list of upscalers to benchmark",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations per method")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per method")
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    log_environment_health()
    _, frame = load_reference_frame(args.input)

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if not methods:
        raise SystemExit("At least one method is required for benchmarking")

    results = []
    for method in methods:
        logger.info("benchmark start method=%s", method)
        try:
            result = benchmark_upscaler(
                method,
                frame,
                iterations=args.iterations,
                warmup_iterations=args.warmup,
            )
        except Exception as exc:
            logger.exception("benchmark method=%s failed: %s", method, exc)
            continue
        results.append(result)

    if not results:
        raise SystemExit("No benchmark results were produced")

    logger.info("benchmark completed methods=%d", len(results))
    for result in results:
        logger.info("benchmark result %s", asdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
