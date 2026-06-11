from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from ...logging_controller import get_logger, setup_logging
from ..ben2.build_all import main as ben2_build_all_main
from ..modnet.build_engine import main as modnet_build_main
from ..registry import get_model_spec


logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildStep:
    name: str
    status: str
    error: str | None = None


def _parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must have four comma-separated integers")
    return tuple(parts)  # type: ignore[return-value]


def _parse_sizes(text: str) -> list[int]:
    sizes: list[int] = []
    for part in text.split(","):
        value = int(part.strip())
        if value <= 0:
            raise argparse.ArgumentTypeError("sizes must be positive integers")
        sizes.append(value)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one benchmark size is required")
    return sizes


def _logging_args(args: argparse.Namespace) -> list[str]:
    forwarded = ["--log-level", args.log_level]
    if args.log_file:
        forwarded.extend(["--log-file", args.log_file])
    if args.log_json:
        forwarded.append("--log-json")
    return forwarded


def _default_modnet_onnx_path() -> Path:
    spec = get_model_spec("modnet-trt")
    return Path(spec.metadata["onnx_path"])


def _default_modnet_engine_path() -> Path:
    spec = get_model_spec("modnet-trt")
    return Path(spec.metadata["engine_path"])


def _default_ben2_engine_dir() -> Path:
    spec = get_model_spec("ben2-trt")
    return Path(spec.local_weights or Path(__file__).resolve().parents[1] / "weights" / "ben2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT engines for MODNet and BEN2")
    parser.add_argument("--skip-modnet", action="store_true", help="Skip the MODNet INT8 engine build")
    parser.add_argument("--skip-ben2", action="store_true", help="Skip the BEN2 INT8 shape build-all set")
    parser.add_argument(
        "--calibration-data-dir",
        default=None,
        help="Directory containing representative calibration images for INT8 builds",
    )
    parser.add_argument(
        "--calibration-cache-path",
        default=None,
        help="Optional path to read/write the TensorRT INT8 calibration cache",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=8,
        help="Batch size used by the INT8 calibrator",
    )
    parser.add_argument(
        "--calibration-max-samples",
        type=int,
        default=32,
        help="Maximum number of calibration images to use",
    )
    parser.add_argument(
        "--modnet-onnx-path",
        default=str(_default_modnet_onnx_path()),
        help="MODNet ONNX input file",
    )
    parser.add_argument(
        "--modnet-engine-path",
        default=str(_default_modnet_engine_path()),
        help="Output MODNet TensorRT engine file",
    )
    parser.add_argument("--modnet-model-key", default="modnet-trt", help="Registry key used for auto-fetch metadata")
    parser.add_argument("--modnet-input-name", default="input", help="MODNet ONNX input tensor name")
    parser.add_argument(
        "--modnet-input-shape",
        default="1,3,512,512",
        type=_parse_shape,
        help="MODNet input shape",
    )
    parser.add_argument("--modnet-workspace-gb", type=int, default=2, help="MODNet TensorRT workspace size in GB")
    parser.add_argument(
        "--modnet-cache-dir",
        default=None,
        help="Optional directory for MODNet timing caches and derived INT8 calibration caches",
    )
    parser.add_argument(
        "--modnet-timing-cache-path",
        default=None,
        help="Optional path for the MODNet TensorRT timing cache",
    )
    parser.add_argument(
        "--modnet-no-auto-fetch-onnx",
        action="store_true",
        help="Disable automatic download of the registry-defined MODNet ONNX file when the file is missing",
    )
    parser.add_argument(
        "--ben2-sizes",
        default="1024,768,512",
        type=_parse_sizes,
        help="Comma-separated square input sizes for BEN2",
    )
    parser.add_argument(
        "--ben2-engine-dir",
        default=str(_default_ben2_engine_dir()),
        help="Directory where BEN2 engines will be written",
    )
    parser.add_argument(
        "--ben2-cache-dir",
        default=None,
        help="Directory for BEN2 TensorRT timing caches",
    )
    parser.add_argument("--ben2-rebuild", action="store_true", help="Rebuild each BEN2 engine even if it already exists")
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def _run_step(name: str, runner, argv: list[str]) -> BuildStep:
    logger.info("trt-build-int8 start step=%s", name)
    try:
        runner(argv)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("trt-build-int8 failed step=%s", name)
        step = BuildStep(name=name, status="failed", error=error)
    else:
        step = BuildStep(name=name, status="ok")
    logger.info("trt-build-int8 result %s", asdict(step))
    return step


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    common_log_args = _logging_args(args)
    steps: list[BuildStep] = []

    if not args.skip_modnet:
        modnet_args = [
            "--onnx-path",
            args.modnet_onnx_path,
            "--engine-path",
            args.modnet_engine_path,
            "--model-key",
            args.modnet_model_key,
            "--input-name",
            args.modnet_input_name,
            "--input-shape",
            ",".join(str(part) for part in args.modnet_input_shape),
            "--workspace-gb",
            str(args.modnet_workspace_gb),
            "--int8",
            "--calibration-batch-size",
            str(args.calibration_batch_size),
            "--calibration-max-samples",
            str(args.calibration_max_samples),
        ]
        if args.calibration_data_dir:
            modnet_args.extend(["--calibration-data-dir", args.calibration_data_dir])
        if args.calibration_cache_path:
            modnet_args.extend(["--calibration-cache-path", args.calibration_cache_path])
        if args.modnet_cache_dir:
            modnet_args.extend(["--cache-dir", args.modnet_cache_dir])
        if args.modnet_timing_cache_path:
            modnet_args.extend(["--timing-cache-path", args.modnet_timing_cache_path])
        if args.modnet_no_auto_fetch_onnx:
            modnet_args.append("--no-auto-fetch-onnx")
        modnet_args.extend(common_log_args)
        steps.append(_run_step("modnet-build", modnet_build_main, modnet_args))

    if not args.skip_ben2:
        ben2_args = [
            "--sizes",
            ",".join(str(size) for size in args.ben2_sizes),
            "--int8",
            "--calibration-batch-size",
            str(args.calibration_batch_size),
            "--calibration-max-samples",
            str(args.calibration_max_samples),
        ]
        if args.calibration_data_dir:
            ben2_args.extend(["--calibration-data-dir", args.calibration_data_dir])
        if args.calibration_cache_path:
            ben2_args.extend(["--calibration-cache-path", args.calibration_cache_path])
        if args.ben2_engine_dir:
            ben2_args.extend(["--engine-dir", args.ben2_engine_dir])
        if args.ben2_cache_dir:
            ben2_args.extend(["--cache-dir", args.ben2_cache_dir])
        if args.ben2_rebuild:
            ben2_args.append("--rebuild")
        ben2_args.extend(common_log_args)
        steps.append(_run_step("ben2-build-all", ben2_build_all_main, ben2_args))

    if not steps:
        raise SystemExit("No TensorRT build steps were selected")

    failed = [step for step in steps if step.status == "failed"]
    logger.info("trt-build-int8 completed steps=%d failed=%d", len(steps), len(failed))
    if failed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
