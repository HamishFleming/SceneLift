# MODNet End-to-End Pipeline

The repository now includes a concrete MODNet workflow:

1. Download the ONNX weights from `Xenova/modnet`.
2. Build a TensorRT engine from the ONNX file.
3. Run the TensorRT engine in a long-lived webcam process.

## Export

Download the Xenova ONNX file:

```bash
modnet-fetch --output-path src/bgremoval/models/weights/modnet/onnx/model.onnx --variant fp32
```

## Build

```bash
modnet-build --onnx-path src/bgremoval/models/weights/modnet/onnx/model.onnx --engine-path src/bgremoval/models/weights/modnet/modnet.engine --fp16
```

If the ONNX file is missing, `modnet-build` will now fetch `Xenova/modnet` automatically unless you disable it:

```bash
modnet-build --onnx-path src/bgremoval/models/weights/modnet/onnx/model.onnx --engine-path src/bgremoval/models/weights/modnet/modnet.engine --no-auto-fetch-onnx
```

Repeated builds are faster when the TensorRT timing cache is reused. By default the build step reads and writes a cache next to the engine file, for example `src/bgremoval/models/weights/modnet/modnet.engine.timing-cache`. You can override that path with `--timing-cache-path` or disable it entirely with `--no-timing-cache`.
If you want multiple engines or shapes to share a cache directory, use `--cache-dir` instead of a single cache file path.
If you are setting up both TensorRT backends at once, `bgremoval-trt-build-all` will run the default MODNet build and the BEN2 build-all pass in one command.

## INT8 build

If you have representative calibration frames, you can build an INT8 engine:

```bash
modnet-build \
  --onnx-path src/bgremoval/models/weights/modnet/onnx/model.onnx \
  --engine-path src/bgremoval/models/weights/modnet/modnet.engine \
  --int8 \
  --calibration-data-dir input/calibration/modnet
```

The builder writes an INT8 calibration cache next to the engine by default, for example `src/bgremoval/models/weights/modnet/modnet.engine.int8.cache`. Use `--calibration-cache-path` if you want to store it elsewhere.

## Runtime

```bash
modnet-run --engine-path src/bgremoval/models/weights/modnet/modnet.engine --mode loopback
```

`--engine-path` must point at a TensorRT engine file such as `modnet.engine`, not the source ONNX file.
The loopback runtime now uses `pyvirtualcam` for the virtual-camera sink, matching the top-level `bgremoval --live` path instead of relying on OpenCV's direct V4L2 writer.

For OBS or any virtual-camera workflow, the top-level CLI uses the same live pipeline:

```bash
bgremoval --input camera:0 --output virtualcam --method modnet --live --live-max-dimension 960
```

If you want to target a custom engine or shape, switch to `modnet-trt` and pass the engine path explicitly:

```bash
bgremoval --input camera:0 --output virtualcam --method modnet-trt --engine-path src/bgremoval/models/weights/modnet/modnet.engine --width 512 --height 512 --live
```

If you are trying to push live throughput much higher, rebuild MODNet at a smaller TensorRT input shape such as `256x256` and run the live path with matching `--width` and `--height`. That is the first practical lever for moving from mid-teens FPS toward higher live rates.

To run a different engine file, point `--engine-path` at it directly:

```bash
modnet-run --engine-path /path/to/custom-modnet.engine --width 512 --height 512 --mode loopback
```

Or stream to SRT:

```bash
modnet-run --engine-path src/bgremoval/models/weights/modnet/modnet.engine --mode srt --srt-url "srt://host:9000?mode=caller&latency=200000"
```

The SRT path is also exposed as a dedicated zero-copy-style entry point:

```bash
modnet-zero-copy --engine-path src/bgremoval/models/weights/modnet/modnet.engine --srt-url "srt://host:9000?mode=caller&latency=200000"
```

## Package integration

- The shared model registry now knows about `modnet-trt`.
- The top-level `bgremoval` CLI also accepts `--method modnet` as a shorthand for `modnet-trt`.
- `modnet-run` uses the engine file passed via `--engine-path`; the top-level `bgremoval` CLI still uses the registry default engine path for `--method modnet` unless you call the model-specific runtime directly.
- The TensorRT helper layer lives in `src/bgremoval/models/tensorrt/`.
- The ONNX export entry point now lives in `src/bgremoval/models/exporters/modnet.py`; `src/bgremoval/models/modnet/export_onnx.py` remains as a compatibility shim.
- The runtime stays local and loads the engine once.
- The loopback runtime requires `pyvirtualcam` plus a writable `v4l2loopback` device. It now uses the same virtual-camera sink path as the top-level live CLI, which is more reliable here than OpenCV's direct V4L2 writer.
- The TensorRT session now creates and owns the active PyCUDA device context before it allocates buffers or streams.
- The TensorRT session now destroys TensorRT objects while the CUDA context is still current, then detaches the CUDA context on shutdown so healthcheck and benchmark runs exit cleanly.
- `bgremoval-healthcheck` now closes the remover on success as well as failure, so a successful TensorRT healthcheck does not leave the CUDA context alive until process exit.
- If a fixed-shape TensorRT engine file was built for a different shape, runtime loading now fails with a clear shape-mismatch error instead of a broadcast crash. Rebuild the engine for the requested size before rerunning.
- The TensorRT session checks the engine's baked input shape before allocating buffers, so a mislabeled fixed-shape engine is rejected early with the shape that needs rebuilding.
- The TensorRT session resolves single-input engine tensor names from the engine itself before calling `set_input_shape`, which avoids failures when the runtime tensor name differs from the ONNX name used at export time.
- On TensorRT 11 builds that do not expose `BuilderFlag.FP16`, the build step will log a warning and continue without the explicit FP16 flag.
- Timing caches are persisted next to the engine by default to reduce rebuild time.
- `modnet-build` accepts `--cache-dir` for shared timing-cache storage.
- Xenova/modnet is the preferred ONNX source for this path.
- The build step now reads the source repo and ONNX filename from the model registry metadata.
- If your target is a true `60 FPS` virtual camera, the current `512x512` MODNet TensorRT path may still be too heavy on this hardware. The next steps are a smaller MODNet engine shape or a lighter backend such as `mediapipe-selfie-segmentation`.
