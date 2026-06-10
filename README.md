# Background Removal Prototype

CLI prototype for testing background removal methods on:

- file input: image or video
- camera input: live webcam stream
- file output: image or video
- loopback output: virtual camera, when `pyvirtualcam` is installed

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

## Notes

- `grabcut` works out of the box and is a useful baseline.
- `rembg` is the AI-backed path and becomes available after installing the optional extra.
- `birefnet` loads `ZhengPeng7/BiRefNet` from Hugging Face with `transformers`, `einops`, `kornia`, `timm`, and `torchvision`.
- `u2net-human-seg` loads `BritishWerewolf/U-2-Net-Human-Seg` from Hugging Face and uses the same HF extra.
- `modnet-trt` is the TensorRT-backed live path for MODNet-style webcam removal.
- `ben2-trt` is the BEN2 ONNX/TensorRT scaffold registered from `onnx-community/BEN2-ONNX`.
- `virtualcam` output requires `pyvirtualcam` plus a system virtual camera backend.
- `--live` enables the low-latency webcam pipeline with frame dropping and resolution control.
- Logging supports console output, `--log-file`, and `--log-json`.
- The model registry lives under `src/bgremoval/models/` and is the place to add new AI backends and their weights.
- MODNet TensorRT now uses `Xenova/modnet` as the ONNX source; use `modnet-fetch` before `modnet-build`.
- `modnet-build` will auto-fetch the Xenova ONNX file if it is missing unless you disable that with `--no-auto-fetch-onnx`.
- `ben2-build` will auto-fetch `onnx-community/BEN2-ONNX` when the ONNX file is missing.
- Canonical live backend: `modnet-trt` on TensorRT.
