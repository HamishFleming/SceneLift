# INT8 Quantization

The TensorRT build path now supports optional INT8 calibration for both `modnet-trt` and `ben2-trt`.

## Supported workflow

- Build from ONNX as usual.
- Add `--int8` to enable calibration-backed engine generation.
- Provide representative frames with `--calibration-data-dir`, or reuse a previous cache with `--calibration-cache-path`.
- Reuse the normal TensorRT timing cache separately from the INT8 calibration cache.
- Use `bgremoval-calibration-set` to turn a larger frame folder into a dedicated calibration directory before building.

## Example

```bash
modnet-build \
  --onnx-path src/bgremoval/models/weights/modnet/onnx/model.onnx \
  --engine-path src/bgremoval/models/weights/modnet/modnet.engine \
  --int8 \
  --calibration-data-dir input/calibration/modnet
```

For BEN2:

```bash
ben2-build \
  --int8 \
  --calibration-data-dir input/calibration/ben2
```

## Calibration cache

The default calibration cache path is derived from the engine path:

- `modnet.engine.int8.cache`
- `ben2.engine.int8.cache`
- `ben2-768.engine.int8.cache`

You can override that with `--calibration-cache-path`.

## Notes

- The calibrator uses resized, normalized calibration images that match the model input shape.
- If you only have a prior calibration cache, the builder can reuse it without new images.
- FP16 remains the default and is still the safest fallback when you do not have good calibration data.
