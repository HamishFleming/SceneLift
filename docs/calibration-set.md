# Calibration Set Helper

Use `bgremoval-calibration-set` to turn a folder of webcam frames or sample images into a smaller calibration directory for TensorRT INT8 builds.

## Example

If you already extracted frames from a clip:

```bash
bgremoval-extract-frames --input input/tam.mp4 --output-dir frames --format png
bgremoval-calibration-set --input-dir frames --output-dir input/calibration/modnet --max-samples 32
```

If you already have a folder of screenshots or webcam captures, point the command at that folder directly:

```bash
bgremoval-calibration-set --input-dir input/samples --output-dir input/calibration/ben2 --max-samples 32
```

## Behavior

- The command scans for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp` files.
- By default it copies up to 32 images into the output directory.
- Output files are named `calib_000000.jpg`, `calib_000001.png`, and so on, preserving the source extension.
- Use `--recursive` if your source images are spread across nested folders.

## Why this exists

The TensorRT INT8 builder needs representative images. This helper gives you a small, dedicated calibration folder that is easy to point at with `--calibration-data-dir` for `modnet-build`, `ben2-build`, `ben2-benchmark`, or `ben2-build-all`.
