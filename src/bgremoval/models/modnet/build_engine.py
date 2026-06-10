from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ...logging_controller import get_logger
from ..hf_onnx import HFOnnxDownloadConfig, download_hf_onnx
from ..registry import get_model_spec
from ..tensorrt.session import save_engine


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModNetEngineConfig:
    onnx_path: Path
    engine_path: Path
    timing_cache_path: Path | None = None
    cache_dir: Path | None = None
    model_key: str = "modnet-trt"
    input_name: str = "input"
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512)
    workspace_size: int = 2 << 30
    fp16: bool = True
    use_timing_cache: bool = True
    auto_fetch_onnx: bool = True


def _parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("input shape must have four comma-separated integers")
    return tuple(parts)  # type: ignore[return-value]


def _parser_error_text(parser) -> str:
    messages: list[str] = []
    for index in range(getattr(parser, "num_errors", 0)):
        error = parser.get_error(index)
        desc = getattr(error, "desc", None)
        if callable(desc):
            try:
                desc = desc()
            except TypeError:
                desc = None
        if desc is None:
            desc = str(error)
        messages.append(str(desc))
    return "\n".join(messages)


def _default_timing_cache_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(engine_path.suffix + ".timing-cache")


def _resolve_timing_cache_path(config: ModNetEngineConfig) -> Path:
    if config.timing_cache_path is not None:
        return config.timing_cache_path
    if config.cache_dir is not None:
        return config.cache_dir / f"{config.engine_path.name}.timing-cache"
    return _default_timing_cache_path(config.engine_path)


def build_engine_from_onnx(config: ModNetEngineConfig) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is required to build an engine. Install the 'trt' extra.") from exc
    try:
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError("pycuda is required to build an engine. Install the 'trt' extra.") from exc

    if not config.onnx_path.exists():
        if not config.auto_fetch_onnx:
            raise FileNotFoundError(f"ONNX file does not exist: {config.onnx_path}")
        spec = get_model_spec(config.model_key)
        source_repo = spec.metadata.get("source_repo")
        source_onnx = spec.metadata.get("source_onnx")
        source_revision = spec.metadata.get("source_revision", "main")
        if not source_repo or not source_onnx:
            raise RuntimeError(f"Registry model {config.model_key} does not define source_repo/source_onnx")
        logger.info("ONNX file missing, fetching %s weights first", config.model_key)
        download_hf_onnx(
            HFOnnxDownloadConfig(
                output_path=config.onnx_path,
                repo_id=str(source_repo),
                filename=str(source_onnx),
                revision=str(source_revision),
            )
        )

    logger.info("Building TensorRT engine from %s", config.onnx_path)
    cuda.init()
    if cuda.Device.count() <= 0:
        raise RuntimeError("No CUDA devices are available for TensorRT engine building")
    cuda_context = cuda.Device(0).make_context()
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_size)
        if config.fp16:
            fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
            if fp16_flag is not None:
                builder_config.set_flag(fp16_flag)
                logger.info("Requested FP16 engine build")
            else:
                logger.warning(
                    "TensorRT %s does not expose BuilderFlag.FP16 in this environment; continuing without the explicit FP16 flag",
                    getattr(trt, "__version__", "unknown"),
                )

        timing_cache_path = _resolve_timing_cache_path(config)
        if config.use_timing_cache:
            if timing_cache_path.exists():
                try:
                    timing_cache = builder_config.create_timing_cache(timing_cache_path.read_bytes())
                    builder_config.set_timing_cache(timing_cache, True)
                    logger.info("Loaded TensorRT timing cache from %s", timing_cache_path)
                except Exception as exc:
                    logger.warning("Could not load TensorRT timing cache %s: %s", timing_cache_path, exc)
            else:
                logger.info("No TensorRT timing cache found at %s; build will populate one", timing_cache_path)

        if not parser.parse(config.onnx_path.read_bytes()):
            errors = _parser_error_text(parser)
            logger.error("TensorRT ONNX parsing failed for %s\n%s", config.onnx_path, errors)
            if config.model_key == "ben2-trt":
                raise RuntimeError(
                    "BEN2 ONNX parsing failed in TensorRT. This model currently appears incompatible with the local "
                    "TensorRT parser/build path in this environment. Try a different TensorRT version or use the ONNX "
                    "artifact directly with an ONNXRuntime-based runtime instead."
                ) from None
            raise RuntimeError(f"TensorRT ONNX parse failed for {config.onnx_path}:\n{errors}")

        profile = builder.create_optimization_profile()
        profile.set_shape(config.input_name, config.input_shape, config.input_shape, config.input_shape)
        builder_config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        engine_path = save_engine(bytes(serialized), config.engine_path)

        if config.use_timing_cache:
            try:
                timing_cache = builder_config.get_timing_cache()
                if timing_cache is None:
                    logger.warning("TensorRT did not return a timing cache after building %s", config.engine_path)
                else:
                    serialized_cache = timing_cache.serialize()
                    timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    timing_cache_path.write_bytes(bytes(serialized_cache))
                    logger.info("Saved TensorRT timing cache to %s", timing_cache_path)
            except Exception as exc:
                logger.warning("Could not save TensorRT timing cache %s: %s", timing_cache_path, exc)

        return engine_path
    finally:
        try:
            cuda_context.pop()
        except Exception:
            logger.debug("TensorRT build CUDA context was already released")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX")
    parser.add_argument("--onnx-path", required=True, help="Input ONNX file")
    parser.add_argument("--engine-path", required=True, help="Output TensorRT engine file")
    parser.add_argument("--model-key", default="modnet-trt", help="Registry key used for auto-fetch metadata")
    parser.add_argument("--input-name", default="input", help="ONNX input tensor name")
    parser.add_argument("--input-shape", default="1,3,512,512", type=_parse_shape, help="Input shape")
    parser.add_argument("--workspace-gb", type=int, default=2, help="TensorRT workspace size in GB")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 if supported")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument(
        "--timing-cache-path",
        default=None,
        help="Optional path for the TensorRT timing cache used to speed up repeated builds",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for timing caches; each engine writes its own cache file there",
    )
    parser.add_argument(
        "--no-timing-cache",
        action="store_true",
        help="Disable loading and saving TensorRT timing caches during the build",
    )
    parser.add_argument(
        "--no-auto-fetch-onnx",
        action="store_true",
        help="Disable automatic download of the registry-defined ONNX file when the file is missing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fp16 = args.fp16 or not args.no_fp16
    build_engine_from_onnx(
        ModNetEngineConfig(
            onnx_path=Path(args.onnx_path),
            engine_path=Path(args.engine_path),
            model_key=args.model_key,
            input_name=args.input_name,
            input_shape=args.input_shape,
            workspace_size=args.workspace_gb << 30,
            fp16=fp16,
            timing_cache_path=Path(args.timing_cache_path) if args.timing_cache_path else None,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            use_timing_cache=not args.no_timing_cache,
            auto_fetch_onnx=not args.no_auto_fetch_onnx,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
