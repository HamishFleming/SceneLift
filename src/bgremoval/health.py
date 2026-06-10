from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .io import open_capture, parse_source
from .logging_controller import get_logger
from .methods import create_remover


logger = get_logger(__name__)


@dataclass(frozen=True)
class HealthCheckResult:
    component: str
    status: str
    elapsed_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    status: str
    load_ms: float
    warmup_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    fps: float
    iterations: int
    warmup_iterations: int
    error: str | None = None


def _close_remover(remover: Any) -> None:
    close = getattr(remover, "close", None)
    if callable(close):
        close()


def log_environment_health() -> None:
    logger.info(
        "healthcheck component=environment status=ok python=%s platform=%s opencv=%s",
        platform.python_version(),
        platform.platform(),
        cv2.__version__,
    )


def _eager_backend_load(remover: Any) -> None:
    probe = getattr(remover, "_get_session", None)
    if callable(probe):
        probe()
        return

    probe = getattr(remover, "_load_session", None)
    if callable(probe):
        probe()
        return

    probe = getattr(remover, "_load_model", None)
    if callable(probe):
        probe()
        return


def load_reference_frame(input_source: str) -> tuple[Any, np.ndarray]:
    source = parse_source(input_source)
    if source.kind == "image":
        frame = cv2.imread(str(source.value), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image input: {source.value}")
        logger.info("healthcheck component=input status=ok kind=image path=%s shape=%s", source.value, frame.shape)
        return source, frame

    capture = open_capture(source)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame from input source: {source.value}")
        logger.info(
            "healthcheck component=input status=ok kind=%s source=%s shape=%s",
            source.kind,
            source.value,
            frame.shape,
        )
        return source, frame
    finally:
        capture.release()


def run_healthcheck(input_source: str, method: str) -> tuple[HealthCheckResult, HealthCheckResult]:
    log_environment_health()
    _, frame = load_reference_frame(input_source)

    start = time.perf_counter()
    remover = None
    try:
        remover = create_remover(method)
        _eager_backend_load(remover)
        load_elapsed = (time.perf_counter() - start) * 1000.0
        logger.info(
            "healthcheck component=backend status=ok method=%s loader=%s load_ms=%.2f",
            method,
            type(remover).__name__,
            load_elapsed,
        )
    except Exception as exc:
        load_elapsed = (time.perf_counter() - start) * 1000.0
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("healthcheck component=backend status=failed method=%s load_ms=%.2f", method, load_elapsed)
        return (
            HealthCheckResult("backend", "failed", load_elapsed, error=error, details={"method": method}),
            HealthCheckResult("inference", "skipped", error=error, details={"method": method}),
        )

    start = time.perf_counter()
    try:
        rgba = remover.remove(frame.copy())
        infer_elapsed = (time.perf_counter() - start) * 1000.0
        logger.info(
            "healthcheck component=inference status=ok method=%s elapsed_ms=%.2f output_shape=%s",
            method,
            infer_elapsed,
            getattr(rgba, "shape", None),
        )
        return (
            HealthCheckResult("backend", "ok", load_elapsed, details={"method": method, "loader": type(remover).__name__}),
            HealthCheckResult(
                "inference",
                "ok",
                infer_elapsed,
                details={"method": method, "output_shape": getattr(rgba, "shape", None)},
            ),
        )
    except Exception as exc:
        infer_elapsed = (time.perf_counter() - start) * 1000.0
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("healthcheck component=inference status=failed method=%s elapsed_ms=%.2f", method, infer_elapsed)
        return (
            HealthCheckResult("backend", "ok", load_elapsed, details={"method": method, "loader": type(remover).__name__}),
            HealthCheckResult("inference", "failed", infer_elapsed, error=error, details={"method": method}),
        )
    finally:
        _close_remover(remover)


def benchmark_method(
    method: str,
    frame: np.ndarray,
    *,
    iterations: int = 10,
    warmup_iterations: int = 2,
) -> BenchmarkResult:
    load_start = time.perf_counter()
    try:
        remover = create_remover(method)
        _eager_backend_load(remover)
        load_ms = (time.perf_counter() - load_start) * 1000.0
    except Exception as exc:
        load_ms = (time.perf_counter() - load_start) * 1000.0
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("benchmark method=%s status=skipped load_ms=%.2f", method, load_ms)
        return BenchmarkResult(
            method=method,
            status="skipped",
            load_ms=load_ms,
            warmup_ms=0.0,
            avg_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            fps=0.0,
            iterations=0,
            warmup_iterations=warmup_iterations,
            error=error,
        )
    return benchmark_remover(
        remover,
        frame,
        method=method,
        load_ms=load_ms,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )


def benchmark_remover(
    remover: Any,
    frame: np.ndarray,
    *,
    method: str,
    load_ms: float,
    iterations: int = 10,
    warmup_iterations: int = 2,
) -> BenchmarkResult:
    try:
        warmup_start = time.perf_counter()
        for _ in range(max(0, warmup_iterations)):
            remover.remove(frame.copy())
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

        timings: list[float] = []
        for _ in range(max(1, iterations)):
            iter_start = time.perf_counter()
            remover.remove(frame.copy())
            timings.append((time.perf_counter() - iter_start) * 1000.0)

        avg_ms = sum(timings) / len(timings)
        min_ms = min(timings)
        max_ms = max(timings)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        logger.info(
            "benchmark method=%s status=ok load_ms=%.2f warmup_ms=%.2f avg_ms=%.2f min_ms=%.2f max_ms=%.2f fps=%.2f iterations=%d warmup=%d",
            method,
            load_ms,
            warmup_ms,
            avg_ms,
            min_ms,
            max_ms,
            fps,
            len(timings),
            warmup_iterations,
        )
        return BenchmarkResult(
            method=method,
            status="ok",
            load_ms=load_ms,
            warmup_ms=warmup_ms,
            avg_ms=avg_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            fps=fps,
            iterations=len(timings),
            warmup_iterations=warmup_iterations,
        )
    finally:
        _close_remover(remover)
