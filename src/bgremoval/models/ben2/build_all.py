from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from ...logging_controller import get_logger, setup_logging
from ..modnet.build_engine import ModNetEngineConfig, build_engine_from_onnx
from ..registry import get_model_spec


logger = get_logger(__name__)


@dataclass(frozen=True)
class Ben2BuildResult:
    size: int
    engine_path: str
    built: bool
    build_ms: float
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


def _engine_needs_rebuild(engine_path: Path, size: int) -> bool:
    if not engine_path.exists():
        return True
    try:
        remover = ModNetTensorRTRemover(
            engine_path=engine_path,
            input_size=(size, size),
        )
        session = remover._get_session()
        session.close()
        return False
    except Exception as exc:
        if _is_shape_mismatch_error(exc):
            logger.warning(
                "Existing BEN2 engine %s does not match requested size %d; rebuilding",
                engine_path,
                size,
            )
            return True
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BEN2 TensorRT engines for multiple shapes")
    parser.add_argument(
        "--sizes",
        default="1024,768,512",
        type=_parse_sizes,
        help="Comma-separated square input sizes to build",
    )
    parser.add_argument(
        "--engine-dir",
        default=None,
        help="Directory where shape-specific BEN2 engines will be written",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for TensorRT timing caches; each size writes its own cache file there",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild each engine even if it already exists")
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    spec = get_model_spec("ben2-trt")
    engine_dir = Path(args.engine_dir) if args.engine_dir else Path(spec.local_weights or Path(__file__).resolve().parents[1] / "weights" / "ben2")
    engine_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else engine_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_path = cache_dir / "ben2-build-all.timing-cache"

    onnx_path = Path(spec.metadata["onnx_path"])
    results: list[Ben2BuildResult] = []
    for size in args.sizes:
        engine_path = _default_engine_path(engine_dir, size)
        should_build = args.rebuild or _engine_needs_rebuild(engine_path, size)
        logger.info("build-all start size=%d engine=%s should_build=%s", size, engine_path, should_build)
        if not should_build:
            results.append(
                Ben2BuildResult(
                    size=size,
                    engine_path=str(engine_path),
                    built=False,
                    build_ms=0.0,
                    status="skipped",
                )
            )
            logger.info("build-all result %s", asdict(results[-1]))
            continue

        build_start = time.perf_counter()
        try:
            build_engine_from_onnx(
                ModNetEngineConfig(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    timing_cache_path=shared_cache_path,
                    model_key="ben2-trt",
                    input_shape=(1, 3, size, size),
                    cache_dir=cache_dir,
                )
            )
            build_ms = (time.perf_counter() - build_start) * 1000.0
            result = Ben2BuildResult(
                size=size,
                engine_path=str(engine_path),
                built=True,
                build_ms=build_ms,
                status="ok",
            )
        except Exception as exc:
            build_ms = (time.perf_counter() - build_start) * 1000.0
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("build-all size=%d failed", size)
            result = Ben2BuildResult(
                size=size,
                engine_path=str(engine_path),
                built=False,
                build_ms=build_ms,
                status="failed",
                error=error,
            )
        results.append(result)
        logger.info("build-all result %s", asdict(result))

    if not results:
        raise SystemExit("No BEN2 build results were produced")

    logger.info("build-all completed sizes=%d", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
