# TensorRT Live Pipeline

This repository is intended to support a low-latency webcam background-removal path for OBS and live streaming.

The preferred layout is:

1. Export each model once to ONNX.
2. Build a TensorRT engine for the target GPU and precision.
3. Run a long-lived webcam process that loads the engine once and processes frames continuously.

## Why this structure

- The ONNX export is model-specific but hardware-agnostic.
- The TensorRT engine is hardware-specific and should be cached per GPU / precision / input shape.
- The runtime should stay hot and avoid reloading weights or rebuilding the engine.

## Runtime targets

- `v4l2loopback` for local OBS virtual camera integration.
- `ffmpeg` piping for remote or networked transport such as SRT.

## Performance guidance

- Keep the model and execution context alive for the lifetime of the process.
- Reuse pagelocked host buffers and device buffers.
- Bind TensorRT tensor addresses once at startup.
- Create and hold an active CUDA context for the lifetime of the TensorRT session.
- Keep only the newest webcam frame when inference lags behind capture.
- Downscale to a practical live resolution before inference if needed.

## Enhancement order

- If the goal is to help segmentation, do only light pre-processing before background removal, such as mild denoising or exposure normalization.
- If the goal is cosmetic improvement, do the enhancement after background removal on the foreground cutout or final composite.
- Avoid aggressive sharpening, upscaling, or style changes before segmentation, because they can introduce edge artifacts and make the matte worse.
- For live use, prefer the simplest possible pre-processing path and keep heavy enhancement out of the hot path.

## Model layout

Recommended repository structure:

- `src/bgremoval/models/tensorrt/`
- `src/bgremoval/models/tensorrt/build_all.py`
- `src/bgremoval/models/exporters/modnet.py`
- `src/bgremoval/models/modnet/build_engine.py`
- `src/bgremoval/models/modnet/runtime.py`
- `src/bgremoval/models/weights/<model-name>/`

This keeps per-model export, build, and runtime logic separated while still fitting the shared background-removal framework.

## Current fit

The existing Python package already supports:

- multiple model backends
- local model registry entries
- live webcam to virtual camera mode
- file and JSON logging

The repository now includes a TensorRT helper layer and a MODNet scaffold matching the export -> build -> runtime pattern.
The repo also exposes `bgremoval-trt-build-all` to build the default MODNet engine plus the BEN2 shape set in one pass.
It also exposes `bgremoval-trt-build-int8` to build the default INT8 MODNet engine plus the BEN2 INT8 shape set in one pass.

The runtime creates the CUDA context from the first visible CUDA device before deserializing the engine, so the PyCUDA stream and TensorRT execution context share a valid device context.
