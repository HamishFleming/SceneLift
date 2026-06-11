# TensorRT Engine Builds

The repository already has model-specific build commands for the TensorRT path:

- `modnet-build`
- `ben2-build`
- `ben2-build-all`

To make initial setup easier, the repo also exposes a single umbrella command:

- `bgremoval-trt-build-all`

For INT8 setup, there is a dedicated umbrella command as well:

- `bgremoval-trt-build-int8`

## What it builds

By default, `bgremoval-trt-build-all` runs:

- `modnet-build`
- `ben2-build-all`

That gives you the default MODNet engine plus the default BEN2 shape set in one pass.

## Example

```bash
bgremoval-trt-build-all
```

If you only want one side of the setup, you can skip the other:

```bash
bgremoval-trt-build-all --skip-ben2
bgremoval-trt-build-all --skip-modnet
```

For INT8 calibration-backed setup:

```bash
bgremoval-trt-build-int8 --calibration-data-dir input/calibration
```

## Notes

- `modnet-build` auto-fetches the Xenova ONNX source if the file is missing.
- `ben2-build-all` uses the shared TensorRT timing-cache flow and can also run INT8 builds if you pass calibration flags directly to that command.
- The umbrella command is meant for first-time setup on a new GPU or venv; the model-specific commands remain the right choice when you need custom shapes, calibration data, or fine-grained control.
- Any extra flags you pass to `bgremoval-trt-build-all` are forwarded to `modnet-build`. That lets you override `--onnx-path`, `--engine-path`, `--input-shape`, or other MODNet build settings while still running the default BEN2 build-all pass. The wrapper’s own logging flags are handled locally and are not forwarded downstream.
- `bgremoval-trt-build-int8` builds the default MODNet INT8 engine and the BEN2 INT8 shape set in one pass. It takes shared calibration flags and explicit MODNet/BEN2 path overrides when you need them.
