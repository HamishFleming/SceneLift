# RVM Backend

This repository now includes an `rvm` backend for Robust Video Matting.

## What it uses

- Torch Hub source: `PeterL1n/RobustVideoMatting`
- Default model variant: `mobilenetv3`
- Runtime style: PyTorch with recurrent state across successive video frames

## Usage

For the top-level live pipeline:

```bash
bgremoval --input camera:0 --output virtualcam --method rvm --live --virtualcam-device /dev/video11
```

For file inputs:

```bash
bgremoval --input input/a.webp --output output/rvm.png --method rvm
```

## Notes

- The first run may take time because Torch Hub can download the upstream repository code and model weights.
- The backend keeps recurrent state between frames, which is useful for video stability but means it is intended primarily for continuous video streams rather than unrelated one-off images in the same process.
- If the input frame size changes during a stream, the backend resets its recurrent state automatically.
- This path depends on the PyTorch stack already used by the repo's Hugging Face backends.
- The upstream RVM project advertises much higher tensor throughput than the current heavier backends, but actual end-to-end virtual-camera FPS still depends on your GPU, the Python capture/output path, and chosen resolution.
