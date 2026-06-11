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
`ben2-build` targets a `1024x1024` engine by default.
The current `BEN2_Base.onnx` artifact is structurally fixed to `1024x1024` in this TensorRT path, so smaller shape requests such as `512x512` do not produce a valid engine from the shipped ONNX file.

## INT8 build

If you have representative calibration frames, you can build an INT8 engine for BEN2 as well:

```bash
ben2-build --int8 --calibration-data-dir input/calibration/ben2
```

For the shape benchmark and build-all helpers, pass the same flags:

```bash
ben2-build-all --int8 --calibration-data-dir input/calibration/ben2
ben2-benchmark --input input/a.webp --int8 --calibration-data-dir input/calibration/ben2
```

For first-time TensorRT setup across both backends, `bgremoval-trt-build-all` will build the default MODNet engine and then the BEN2 build-all shape set.

The default calibration cache path is derived from the engine filename, so `ben2.engine` writes `ben2.engine.int8.cache` and `ben2-768.engine` writes `ben2-768.engine.int8.cache`.

## Run

```bash
ben2-run --mode loopback
```

For OBS or any virtual-camera workflow, the top-level CLI uses the same live pipeline:

```bash
bgremoval --input camera:0 --output virtualcam --method ben2 --live --live-max-dimension 960
```

If you want to target a specific engine file, switch to `ben2-trt` and pass the engine path explicitly:

```bash
bgremoval --input camera:0 --output virtualcam --method ben2-trt --engine-path src/bgremoval/models/weights/ben2/ben2.engine --width 1024 --height 1024 --live
```

The TensorRT engine file and the live width/height must match the fixed build shape. With the current shipped BEN2 ONNX, that means `1024x1024`.
The top-level `bgremoval` live path now validates the TensorRT engine before opening the capture loop, so shape mismatches fail fast instead of surfacing on the first frame.

To run a specific engine file, pass it explicitly:

```bash
ben2-run --engine-path /path/to/ben2-768.engine --width 768 --height 768 --mode loopback
```

Or use SRT:

```bash
ben2-run --mode srt --srt-url "srt://host:9000?mode=caller&latency=200000"
```

## Notes

- BEN2 is registered in the shared model registry under `ben2-trt`.
- The top-level `bgremoval` CLI also accepts `--method ben2` as a shorthand for `ben2-trt`.
- The model source is `onnx-community/BEN2-ONNX`.
- The ONNX file exposed by the repository is `onnx/model_fp16.onnx`.
- The runtime scaffold uses the shared TensorRT live pipeline and can be tuned further once the ONNX graph is validated end to end.
- In this environment, TensorRT parsing of the BEN2 ONNX graph is currently failing with a shape-tensor compatibility error, so engine generation is experimental until that graph is made TRT-compatible or converted through another path.
- BEN2 performance in the current TensorRT path is constrained by the upstream `1024x1024` ONNX artifact. See [`docs/ben2-performance.md`](/mnt/code/ai/background-removal/ben/docs/ben2-performance.md) for the current limitations.
- `ben2-run` uses the engine file passed via `--engine-path`; the top-level `bgremoval` CLI still uses the registry default engine path for `--method ben2` unless you call the model-specific runtime directly.
- The shared TensorRT session destroys TensorRT objects while the CUDA context is still current, then detaches the CUDA context on shutdown so healthcheck and benchmark runs exit cleanly.
- `bgremoval-healthcheck` closes the remover after a successful BEN2 probe, so a passing healthcheck does not leak a CUDA context until process exit.
- If a shape-specific engine file was built for a different fixed shape, runtime loading now fails with a clear shape-mismatch error instead of a low-level broadcast crash. Rebuild the engine for the requested size before rerunning.
- The TensorRT session checks the engine's baked input shape before allocating buffers, so a mislabeled BEN2 engine is rejected early with the shape that needs rebuilding.
- The default `ben2-benchmark` and `ben2-build-all` flows now target `1024` only, because the shipped BEN2 ONNX artifact does not currently support smaller TensorRT build shapes.
