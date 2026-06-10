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
    stream: Any
    inputs: list[TensorRTIOBinding]
    outputs: list[TensorRTIOBinding]

    @classmethod
    def from_engine_path(cls, engine_path: str | Path, logger_name: str | None = None) -> "TensorRTSession":
        trt = _require_tensorrt()
        cuda = _require_pycuda()

        path = Path(engine_path)
        logger.info("Loading TensorRT engine from %s", path)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with path.open("rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Could not create TensorRT execution context")

        stream = cuda.Stream()
        inputs: list[TensorRTIOBinding] = []
        outputs: list[TensorRTIOBinding] = []

        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = engine.get_tensor_shape(name)
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

        return cls(engine=engine, context=context, stream=stream, inputs=inputs, outputs=outputs)

    def infer(self, input_array) -> Any:
        cuda = _require_pycuda()
        if not self.inputs or not self.outputs:
            raise RuntimeError("TensorRTSession has no bound IO tensors")

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


def load_engine(engine_path: str | Path) -> TensorRTSession:
    return TensorRTSession.from_engine_path(engine_path)


def save_engine(engine_bytes: bytes, engine_path: str | Path) -> Path:
    path = Path(engine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(engine_bytes)
    logger.info("Saved TensorRT engine to %s", path)
    return path
