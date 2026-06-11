from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...logging_controller import get_logger
from ..hf_onnx import HFOnnxDownloadConfig, download_hf_onnx
from ..registry import get_model_spec
from ..tensorrt.calibration import (
    build_calibration_batches,
    collect_image_paths,
    create_image_folder_int8_calibrator,
)
from ..tensorrt.session import save_engine


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModNetEngineConfig:
    onnx_path: Path
    engine_path: Path
    timing_cache_path: Path | None = None
    cache_dir: Path | None = None
    calibration_data_dir: Path | None = None
    calibration_cache_path: Path | None = None
    model_key: str = "modnet-trt"
    input_name: str = "input"
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512)
    workspace_size: int = 2 << 30
    fp16: bool = True
    int8: bool = False
    calibration_batch_size: int = 8
    calibration_max_samples: int = 32
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


def _default_calibration_cache_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(engine_path.suffix + ".int8.cache")


def _resolve_timing_cache_path(config: ModNetEngineConfig) -> Path:
    if config.timing_cache_path is not None:
        return config.timing_cache_path
    if config.cache_dir is not None:
        return config.cache_dir / f"{config.engine_path.name}.timing-cache"
    return _default_timing_cache_path(config.engine_path)


def _resolve_calibration_cache_path(config: ModNetEngineConfig) -> Path:
    if config.calibration_cache_path is not None:
        return config.calibration_cache_path
    if config.cache_dir is not None:
        return config.cache_dir / f"{config.engine_path.name}.int8.cache"
    return _default_calibration_cache_path(config.engine_path)


def _shape_matches_network(engine_shape: tuple[int, ...], requested_shape: tuple[int, ...]) -> bool:
    if len(engine_shape) != len(requested_shape):
        return False
    for engine_dim, requested_dim in zip(engine_shape, requested_shape):
        if engine_dim in (-1, 0):
            continue
        if engine_dim != requested_dim:
            return False
    return True


def _resolve_network_input(network, requested_name: str):
    if network.num_inputs <= 0:
        raise RuntimeError("Parsed ONNX network does not expose any inputs")
    if network.num_inputs == 1:
        network_input = network.get_input(0)
        if network_input.name != requested_name:
            logger.warning(
                "TensorRT parsed network input name %s does not match requested name %s; using parsed input tensor",
                network_input.name,
                requested_name,
            )
        return network_input
    for index in range(network.num_inputs):
        network_input = network.get_input(index)
        if network_input.name == requested_name:
            return network_input
    input_names = [network.get_input(index).name for index in range(network.num_inputs)]
    raise RuntimeError(
        f"TensorRT parsed network input tensor {requested_name!r} was not found. Available network inputs: {input_names}"
    )


def _preprocess_calibration_image(path: Path, input_shape: tuple[int, int, int, int]) -> np.ndarray:
    height = input_shape[2]
    width = input_shape[3]
    frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read calibration image: {path}")
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)


def _build_int8_calibrator(config: ModNetEngineConfig):
    if not config.int8:
        return None

    calibration_cache_path = _resolve_calibration_cache_path(config)
    calibration_data_dir = config.calibration_data_dir
    if calibration_data_dir is None and not calibration_cache_path.exists():
        raise RuntimeError(
            "INT8 calibration was requested but no calibration data directory or existing calibration cache was provided. "
            "Pass --calibration-data-dir with representative images or --calibration-cache-path pointing to a prior cache."
        )

    batches: list[np.ndarray] = []
    if calibration_data_dir is not None:
        image_paths = collect_image_paths(calibration_data_dir, max_samples=config.calibration_max_samples)
        if not image_paths and not calibration_cache_path.exists():
            raise RuntimeError(f"No calibration images were found in {calibration_data_dir}")
        if image_paths:
            batches = build_calibration_batches(
                image_paths,
                batch_size=config.calibration_batch_size,
                preprocess=lambda path: _preprocess_calibration_image(path, config.input_shape),
            )
            if not batches and not calibration_cache_path.exists():
                raise RuntimeError(
                    "Calibration images were found, but there were not enough samples to form a full INT8 batch. "
                    f"Increase --calibration-max-samples or reduce --calibration-batch-size (current batch size: {config.calibration_batch_size})."
                )

    if not batches and not calibration_cache_path.exists():
        raise RuntimeError(
            "INT8 calibration needs at least one full batch of representative samples or an existing calibration cache."
        )

    return create_image_folder_int8_calibrator(
        batches,
        batch_size=config.calibration_batch_size,
        cache_path=calibration_cache_path,
    )


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
        calibrator = None
        if config.int8:
            int8_flag = getattr(trt.BuilderFlag, "INT8", None)
            if int8_flag is None:
                raise RuntimeError(f"TensorRT {getattr(trt, '__version__', 'unknown')} does not expose BuilderFlag.INT8")
            builder_config.set_flag(int8_flag)
            calibrator = _build_int8_calibrator(config)
            if hasattr(builder_config, "int8_calibrator"):
                builder_config.int8_calibrator = calibrator
            elif hasattr(builder_config, "set_int8_calibrator"):
                builder_config.set_int8_calibrator(calibrator)
            else:
                raise RuntimeError("TensorRT builder config does not expose an INT8 calibrator setter")
            logger.info(
                "Requested INT8 engine build with calibration cache %s",
                _resolve_calibration_cache_path(config),
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

        network_input = _resolve_network_input(network, config.input_name)
        parsed_input_shape = tuple(int(dim) for dim in network_input.shape)
        requested_shape_override = not _shape_matches_network(parsed_input_shape, config.input_shape)
        if requested_shape_override:
            try:
                network_input.shape = config.input_shape
                logger.info(
                    "Overrode parsed ONNX input shape for %s from %s to %s",
                    network_input.name,
                    parsed_input_shape,
                    config.input_shape,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"TensorRT parsed input tensor {network_input.name!r} has static shape {parsed_input_shape}, "
                    f"but build requested shape {config.input_shape}. The ONNX input shape could not be overridden."
                ) from exc

        profile = builder.create_optimization_profile()
        profile.set_shape(network_input.name, config.input_shape, config.input_shape, config.input_shape)
        builder_config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            if config.model_key == "ben2-trt" and requested_shape_override:
                raise RuntimeError(
                    "BEN2_Base.onnx is structurally fixed to its exported 1024x1024 input layout in this TensorRT path. "
                    f"A build for requested shape {config.input_shape} cannot be produced from this ONNX artifact. "
                    "Use a 1024x1024 BEN2 build, or generate a BEN2-specific ONNX export that was authored for the target size."
                )
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
            int8=args.int8,
            calibration_data_dir=Path(args.calibration_data_dir) if args.calibration_data_dir else None,
            calibration_cache_path=Path(args.calibration_cache_path) if args.calibration_cache_path else None,
            calibration_batch_size=args.calibration_batch_size,
            calibration_max_samples=args.calibration_max_samples,
            timing_cache_path=Path(args.timing_cache_path) if args.timing_cache_path else None,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            use_timing_cache=not args.no_timing_cache,
            auto_fetch_onnx=not args.no_auto_fetch_onnx,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
