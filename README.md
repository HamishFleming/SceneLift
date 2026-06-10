# Background Removal Prototype

CLI prototype for testing background removal methods on:

- file input: image or video
- camera input: live webcam stream
- file output: image or video
- loopback output: virtual camera, when `pyvirtualcam` is installed

The primary CLI is `bgremoval`; `bgremove` is provided as a compatibility alias.

## Install

```bash
pip install -e .
pip install -e ".[ai,virtualcam]"
pip install -e ".[hf]"
```

## Examples

Process an image file to a transparent PNG:

```bash
bgremoval --input samples/person.jpg --output out.png --method grabcut
```

Process a video file to another video file:

```bash
bgremoval --input samples/input.mp4 --output out.mp4 --method rembg
```

Use webcam `0` as input and send to a loopback camera:

```bash
bgremoval --input camera:0 --output virtualcam --method birefnet --live
```

Live mode keeps only the newest frame and is the recommended path for OBS:

```bash
bgremoval --input camera:0 --output virtualcam --method birefnet --live --live-max-dimension 960
```

Benchmark the available backends on the same input frame:

```bash
bgremoval-benchmark --input input/a.webp --methods modnet-trt,u2net-human-seg,mediapipe-selfie-segmentation,grabcut
```

Split a video into a folder of frames:

```bash
bgremoval-extract-frames --input input/tam.mp4 --output-dir frames --format webp
```

Run a startup healthcheck for one backend:

```bash
bgremoval-healthcheck --input input/a.webp --method mediapipe-selfie-segmentation
```

## Notes

- `grabcut` works out of the box and is a useful baseline.
- `rembg` is the AI-backed path and becomes available after installing the optional extra.
- `birefnet` loads `ZhengPeng7/BiRefNet` from Hugging Face with `transformers`, `einops`, `kornia`, `timm`, and `torchvision`.
- `u2net-human-seg` downloads `onnx/model.onnx` and `preprocessor_config.json` from `BritishWerewolf/U-2-Net-Human-Seg` and runs them through ONNXRuntime.
- `mediapipe-selfie-segmentation` downloads `onnx/model.onnx` and `preprocessor_config.json` from `onnx-community/mediapipe_selfie_segmentation` and runs them through ONNXRuntime.
- `modnet-trt` is the TensorRT-backed live path for MODNet-style webcam removal. The CLI also accepts `--method modnet` as a shorthand alias.
- `ben2-trt` is the BEN2 ONNX/TensorRT scaffold registered from `onnx-community/BEN2-ONNX`. The CLI also accepts `--method ben2` as a shorthand alias.
- `virtualcam` output requires `pyvirtualcam` plus a system virtual camera backend.
- `--live` enables the low-latency webcam pipeline with frame dropping and resolution control.
- Backends may run at a smaller internal inference size; the CLI resizes their RGBA output back to the input frame size before writing.
- With `--log-level DEBUG`, the CLI logs when backend output is resized to match the source frame.
- Logging supports console output, `--log-file`, and `--log-json`.
- Benchmarking and healthcheck commands share the same structured logging.
- `bgremoval-extract-frames` writes sequentially numbered `frame_000000.png` or `.webp` files to a target directory.
- The model registry lives under `src/bgremoval/models/` and is the place to add new AI backends and their weights.
- MODNet TensorRT now uses `Xenova/modnet` as the ONNX source; use `modnet-fetch` before `modnet-build`.
- `modnet-build` will auto-fetch the Xenova ONNX file if it is missing unless you disable that with `--no-auto-fetch-onnx`.
- `ben2-build` will auto-fetch `onnx-community/BEN2-ONNX` when the ONNX file is missing.
- Canonical live backend: `modnet-trt` on TensorRT.
