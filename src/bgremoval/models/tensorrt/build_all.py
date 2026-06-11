from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from ...logging_controller import get_logger, setup_logging
from ..ben2.build_all import main as ben2_build_all_main
from ..modnet.build_engine import main as modnet_build_main


logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildStep:
    name: str
    status: str
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the default TensorRT engine set for the repo")
    parser.add_argument("--skip-modnet", action="store_true", help="Skip the default MODNet TensorRT build")
    parser.add_argument("--skip-ben2", action="store_true", help="Skip the BEN2 TensorRT build-all set")
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def _run_step(name: str, runner, forwarded_args: list[str]) -> BuildStep:
    logger.info("trt-build-all start step=%s", name)
    try:
        runner(forwarded_args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("trt-build-all failed step=%s", name)
        step = BuildStep(name=name, status="failed", error=error)
    else:
        step = BuildStep(name=name, status="ok")
    logger.info("trt-build-all result %s", asdict(step))
    return step


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, modnet_extra_args = parser.parse_known_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    steps: list[BuildStep] = []
    if not args.skip_modnet:
        if modnet_extra_args:
            logger.info("Forwarding extra arguments to modnet-build: %s", modnet_extra_args)
        steps.append(_run_step("modnet-build", modnet_build_main, modnet_extra_args))
    if not args.skip_ben2:
        steps.append(_run_step("ben2-build-all", ben2_build_all_main, []))

    if not steps:
        raise SystemExit("No TensorRT build steps were selected")

    failed = [step for step in steps if step.status == "failed"]
    logger.info("trt-build-all completed steps=%d failed=%d", len(steps), len(failed))
    if failed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
