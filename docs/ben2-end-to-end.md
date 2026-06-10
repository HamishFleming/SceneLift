# BEN2 End-to-End Pipeline

`onnx-community/BEN2-ONNX` is now registered as `ben2-trt`, with matching fetch/build/run entry points.

## Fetch ONNX

```bash
ben2-fetch
```

This downloads `onnx/model_fp16.onnx` from `onnx-community/BEN2-ONNX` into:

```text
src/bgremoval/models/weights/ben2/onnx/model_fp16.onnx
```

## Build TensorRT engine

```bash
ben2-build
```

The build step is registry-driven and uses the BEN2 source metadata to auto-fetch the ONNX file if it is missing.
It also requests FP16 directly instead of relying on a TensorRT Python capability attribute that is not present in some builds.

## Run

```bash
ben2-run --mode loopback
```

Or use SRT:

```bash
ben2-run --mode srt --srt-url "srt://host:9000?mode=caller&latency=200000"
```

## Notes

- BEN2 is registered in the shared model registry under `ben2-trt`.
- The model source is `onnx-community/BEN2-ONNX`.
- The ONNX file exposed by the repository is `onnx/model_fp16.onnx`.
- The runtime scaffold uses the shared TensorRT live pipeline and can be tuned further once the ONNX graph is validated end to end.
- In this environment, TensorRT parsing of the BEN2 ONNX graph is currently failing with a shape-tensor compatibility error, so engine generation is experimental until that graph is made TRT-compatible or converted through another path.
