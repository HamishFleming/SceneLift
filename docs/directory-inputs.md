# Directory Inputs

`bgremoval` can process a directory of image frames as input.

Example:

```bash
bgremoval --input input/frames --output output/frames/modnet-trt --method modnet-trt --engine-path src/bgremoval/models/weights/modnet/modnet.engine
```

Behavior:

- The input path must be a directory containing image files.
- Files are processed in sorted path order.
- Supported input images are the same as the rest of the CLI: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, and `.tiff`.
- Output is written to the target directory as `.png` files, one per readable input frame.
- Unreadable files are skipped with a warning.

This mode is useful when you have already extracted frames from a video and want to run a TensorRT backend over the frame folder directly without re-wrapping the frames into a video container first.
