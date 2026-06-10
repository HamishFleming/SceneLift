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
- Keep only the newest webcam frame when inference lags behind capture.
- Downscale to a practical live resolution before inference if needed.

## Model layout

Recommended repository structure:

- `src/bgremoval/models/tensorrt/`
- `src/bgremoval/models/modnet/export_onnx.py`
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
