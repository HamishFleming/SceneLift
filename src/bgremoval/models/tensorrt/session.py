from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...logging_controller import get_logger


logger = get_logger(__name__)


def _require_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is not installed. Install the 'trt' extra first.") from exc
    return trt


def _require_pycuda():
    try:
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError("pycuda is not installed. Install the 'trt' extra first.") from exc
    return cuda


def _shape_matches_engine(engine_shape: tuple[int, ...], requested_shape: tuple[int, ...]) -> bool:
    if len(engine_shape) != len(requested_shape):
        return False
    for engine_dim, requested_dim in zip(engine_shape, requested_shape):
        if engine_dim in (-1, 0):
            continue
        if engine_dim != requested_dim:
            return False
    return True


@dataclass(frozen=True)
class TensorRTIOBinding:
    name: str
    host: Any
    device: Any
    mode: Any


@dataclass
class TensorRTSession:
    engine: Any
    context: Any
    cuda_context: Any
    stream: Any
    inputs: list[TensorRTIOBinding]
    outputs: list[TensorRTIOBinding]
    closed: bool = False

    def _release_cuda_context(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.cuda_context.push()
        except Exception:
            logger.debug("TensorRT CUDA context push during cleanup was skipped")
        try:
            # Drop TensorRT objects while the CUDA context is still current.
            self.outputs = []
            self.inputs = []
            self.stream = None
            self.context = None
            self.engine = None
        finally:
            try:
                self.cuda_context.pop()
            except Exception:
                logger.debug("TensorRT CUDA context was already released")
        try:
            detach = getattr(self.cuda_context, "detach", None)
            if callable(detach):
                detach()
        except Exception:
            logger.debug("TensorRT CUDA context detach was already skipped")

    @classmethod
    def from_engine_path(
        cls,
        engine_path: str | Path,
        logger_name: str | None = None,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> "TensorRTSession":
        trt = _require_tensorrt()
        cuda = _require_pycuda()

        path = Path(engine_path)
        logger.info("Loading TensorRT engine from %s", path)
        cuda.init()
        if cuda.Device.count() <= 0:
            raise RuntimeError("No CUDA devices are available for TensorRT execution")
        device = cuda.Device(0)
        cuda_context = device.make_context()
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with path.open("rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            if engine is None:
                raise RuntimeError(f"Could not deserialize TensorRT engine: {path}")

            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError("Could not create TensorRT execution context")

            engine_input_names: list[str] = []
            engine_tensor_names: list[str] = []
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                engine_tensor_names.append(name)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    engine_input_names.append(name)

            if input_shapes:
                resolved_input_shapes = dict(input_shapes)
                if len(input_shapes) == 1 and len(engine_input_names) == 1:
                    requested_name, shape = next(iter(input_shapes.items()))
                    engine_input_name = engine_input_names[0]
                    if requested_name != engine_input_name:
                        logger.warning(
                            "TensorRT engine input name %s does not match requested name %s; using engine tensor name",
                            engine_input_name,
                            requested_name,
                        )
                        resolved_input_shapes = {engine_input_name: shape}
                for name, shape in resolved_input_shapes.items():
                    if name not in engine_tensor_names:
                        if len(engine_input_names) == 1:
                            logger.warning(
                                "TensorRT engine does not expose tensor name %s; using sole input tensor %s",
                                name,
                                engine_input_names[0],
                            )
                            name = engine_input_names[0]
                        else:
                            raise RuntimeError(
                                f"TensorRT engine input tensor {name!r} was not found. Available tensors: {engine_tensor_names}"
                            )
                    engine_shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
                    if not _shape_matches_engine(engine_shape, shape):
                        raise RuntimeError(
                            f"TensorRT engine {path} was built for input shape {engine_shape} on tensor {name!r}, "
                            f"but shape {shape} was requested. Rebuild the engine for the requested size before running."
                        )
                set_input_shape = getattr(context, "set_input_shape", None)
                if not callable(set_input_shape):
                    raise RuntimeError("TensorRT execution context does not support set_input_shape")
                for name, shape in resolved_input_shapes.items():
                    try:
                        set_input_shape(name, shape)
                    except Exception as exc:
                        raise RuntimeError(
                            f"TensorRT engine {path} does not accept requested input shape {shape} for tensor {name!r}. "
                            "The engine file may have been built for a different fixed shape and needs to be rebuilt."
                        ) from exc

            stream = cuda.Stream()
            inputs: list[TensorRTIOBinding] = []
            outputs: list[TensorRTIOBinding] = []

            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                shape = context.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                size = trt.volume(shape)
                host = cuda.pagelocked_empty(size, dtype)
                device = cuda.mem_alloc(host.nbytes)
                mode = engine.get_tensor_mode(name)
                binding = TensorRTIOBinding(name=name, host=host, device=device, mode=mode)
                if mode == trt.TensorIOMode.INPUT:
                    inputs.append(binding)
                else:
                    outputs.append(binding)
                context.set_tensor_address(name, int(device))

            return cls(
                engine=engine,
                context=context,
                cuda_context=cuda_context,
                stream=stream,
                inputs=inputs,
                outputs=outputs,
            )
        except Exception:
            try:
                detach = getattr(cuda_context, "detach", None)
                cuda_context.pop()
                if callable(detach):
                    detach()
            except Exception:
                logger.debug("TensorRT CUDA context cleanup after load failure was skipped")
            raise

    def infer(self, input_array) -> Any:
        cuda = _require_pycuda()
        if not self.inputs or not self.outputs:
            raise RuntimeError("TensorRTSession has no bound IO tensors")

        self.cuda_context.push()
        try:
            input_binding = self.inputs[0]
            output_binding = self.outputs[0]
            input_host = input_binding.host
            output_host = output_binding.host

            input_host[:] = input_array.ravel()
            cuda.memcpy_htod_async(input_binding.device, input_host, self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(output_host, output_binding.device, self.stream)
            self.stream.synchronize()
            return output_host
        finally:
            self.cuda_context.pop()

    def close(self) -> None:
        self._release_cuda_context()


def load_engine(
    engine_path: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> TensorRTSession:
    return TensorRTSession.from_engine_path(engine_path, input_shapes=input_shapes)


def save_engine(engine_bytes: bytes, engine_path: str | Path) -> Path:
    path = Path(engine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(engine_bytes)
    logger.info("Saved TensorRT engine to %s", path)
    return path
