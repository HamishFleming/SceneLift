from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from ...health import benchmark_remover, load_reference_frame, log_environment_health
from ...logging_controller import get_logger, setup_logging
from ..registry import get_model_spec
from ..modnet.build_engine import ModNetEngineConfig, build_engine_from_onnx
from ..modnet.trt_backend import ModNetTensorRTRemover


logger = get_logger(__name__)


@dataclass(frozen=True)
class Ben2ShapeBenchmarkResult:
    size: int
    engine_path: str
    built: bool
    build_ms: float
    load_ms: float
    warmup_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    fps: float
    iterations: int
    warmup_iterations: int
    status: str
    error: str | None = None


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


def _default_engine_path(engine_dir: Path, size: int) -> Path:
    if size == 1024:
        return engine_dir / "ben2.engine"
    return engine_dir / f"ben2-{size}.engine"


def _is_shape_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "static dimension mismatch" in message
        or "could not broadcast input array" in message
        or "setinputshape" in message
    )


def _ensure_engine(
    size: int,
    engine_dir: Path,
    rebuild: bool,
    cache_dir: Path | None,
    *,
    int8: bool = False,
    calibration_data_dir: Path | None = None,
    calibration_cache_path: Path | None = None,
    calibration_batch_size: int = 8,
    calibration_max_samples: int = 32,
) -> tuple[Path, float, bool]:
    spec = get_model_spec("ben2-trt")
    engine_path = _default_engine_path(engine_dir, size)
    should_build = rebuild or not engine_path.exists()
    if not should_build:
        try:
            remover = ModNetTensorRTRemover(
                engine_path=engine_path,
                input_size=(size, size),
            )
            session = remover._get_session()
            session.close()
        except Exception as exc:
            if _is_shape_mismatch_error(exc):
                logger.warning(
                    "Existing BEN2 engine %s does not match requested size %d; rebuilding",
                    engine_path,
                    size,
                )
                should_build = True
            else:
                raise
    if not should_build:
        return engine_path, 0.0, False

    build_start = time.perf_counter()
    build_engine_from_onnx(
        ModNetEngineConfig(
            onnx_path=Path(spec.metadata["onnx_path"]),
            engine_path=engine_path,
            model_key="ben2-trt",
            input_shape=(1, 3, size, size),
            cache_dir=cache_dir,
            int8=int8,
            calibration_data_dir=calibration_data_dir,
            calibration_cache_path=calibration_cache_path,
            calibration_batch_size=calibration_batch_size,
            calibration_max_samples=calibration_max_samples,
        )
    )
    build_ms = (time.perf_counter() - build_start) * 1000.0
    return engine_path, build_ms, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark BEN2 TensorRT engine shapes")
    parser.add_argument("--input", "-i", required=True, help="Input file path or camera spec like camera:0")
    parser.add_argument(
        "--sizes",
        default="1024,768,512",
        type=_parse_sizes,
        help="Comma-separated square input sizes to benchmark",
    )
    parser.add_argument(
        "--engine-dir",
        default=None,
        help="Directory containing or receiving shape-specific BEN2 engines",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for TensorRT timing caches; each size writes its own cache file there",
    )
    parser.add_argument("--int8", action="store_true", help="Enable INT8 calibration and engine build")
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
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations per size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per size")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild each engine even if it already exists")
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    log_environment_health()
    _, frame = load_reference_frame(args.input)

    spec = get_model_spec("ben2-trt")
    engine_dir = Path(args.engine_dir) if args.engine_dir else Path(spec.local_weights or Path(__file__).resolve().parents[1] / "weights" / "ben2")
    engine_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else engine_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[Ben2ShapeBenchmarkResult] = []
    for size in args.sizes:
        logger.info("shape benchmark start size=%d", size)
        try:
            engine_path, build_ms, built = _ensure_engine(
                size,
                engine_dir,
                args.rebuild,
                cache_dir,
                int8=args.int8,
                calibration_data_dir=Path(args.calibration_data_dir) if args.calibration_data_dir else None,
                calibration_cache_path=Path(args.calibration_cache_path) if args.calibration_cache_path else None,
                calibration_batch_size=args.calibration_batch_size,
                calibration_max_samples=args.calibration_max_samples,
            )
            load_start = time.perf_counter()
            remover = ModNetTensorRTRemover(
                engine_path=engine_path,
                input_size=(size, size),
            )
            remover._get_session()
            load_ms = (time.perf_counter() - load_start) * 1000.0
            result = benchmark_remover(
                remover,
                frame,
                method=f"ben2-{size}",
                load_ms=load_ms,
                iterations=args.iterations,
                warmup_iterations=args.warmup,
            )
            shape_result = Ben2ShapeBenchmarkResult(
                size=size,
                engine_path=str(engine_path),
                built=built,
                build_ms=build_ms,
                load_ms=result.load_ms,
                warmup_ms=result.warmup_ms,
                avg_ms=result.avg_ms,
                min_ms=result.min_ms,
                max_ms=result.max_ms,
                fps=result.fps,
                iterations=result.iterations,
                warmup_iterations=result.warmup_iterations,
                status=result.status,
                error=result.error,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("shape benchmark size=%d failed", size)
            shape_result = Ben2ShapeBenchmarkResult(
                size=size,
                engine_path=str(_default_engine_path(engine_dir, size)),
                built=False,
                build_ms=0.0,
                load_ms=0.0,
                warmup_ms=0.0,
                avg_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                fps=0.0,
                iterations=0,
                warmup_iterations=args.warmup,
                status="failed",
                error=error,
            )
        results.append(shape_result)
        logger.info("shape benchmark result %s", asdict(shape_result))

    if not results:
        raise SystemExit("No BEN2 benchmark results were produced")

    logger.info("shape benchmark completed sizes=%d", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
