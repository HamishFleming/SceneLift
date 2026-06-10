# Frame Extraction

`bgremoval-extract-frames` writes the frames from a video input into a directory of numbered image files.

Example:

```bash
bgremoval-extract-frames --input input/tam.mp4 --output-dir frames --format png
```

Use `--format webp` if you want smaller files and are fine with lossy compression:

```bash
bgremoval-extract-frames --input input/tam.mp4 --output-dir frames --format webp --webp-quality 90
```

Defaults:

- output filenames use the pattern `frame_000000.png` or `frame_000000.webp`
- numbering starts at `0`
- WebP quality defaults to `90`
- the command stops only when the video ends unless `--max-frames` is set

The command expects a video file input and writes the extracted frames into the target directory, creating it if needed.
