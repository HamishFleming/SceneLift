# Real-ESRGAN Upscaling

The repository now has a dedicated upscaling command family built around Real-ESRGAN:

- `bgremoval-upscale`
- `bgremoval-upscale-benchmark`
- `bgremoval-upscale-healthcheck`

## What it supports

- image files
- folders of images
- video files
- camera inputs
- virtual camera output for OBS-style workflows

## Example

Upscale a single image:

```bash
bgremoval-upscale --input input/photo.jpg --output output/photo-upscaled.png
```

Upscale a folder of frames:

```bash
bgremoval-upscale --input input/frames --output output/frames-upscaled
```

Send a camera feed to a virtual camera at the upscaled resolution:

```bash
bgremoval-upscale --input camera:0 --output virtualcam --method realesrgan-x4plus
```

If you want the highest possible throughput for a live virtual camera, add `--virtualcam-no-sleep` so the pipeline does not wait for the nominal FPS clock:

```bash
bgremoval-upscale --input camera:0 --output virtualcam --method realesrgan-x2plus --virtualcam-no-sleep
```

## Model files

The default registry points at local weights under:

`src/bgremoval/models/weights/realesrgan/`

Recommended filenames:

- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x4plus_anime_6B.pth`
- `RealESRGAN_x2plus.pth`

If you want to use a custom checkpoint, pass `--model-path`.

The `bgremoval-model-pull-all` command mirrors the two release-backed checkpoints automatically:

- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x4plus_anime_6B.pth`

`RealESRGAN_x2plus.pth` is still a valid runtime option, but it remains a manual download for now.

## Dependencies

Install the upscaling extra before using these commands:

```bash
pip install -e '.[upscale]'
```

The backend lazily imports `realesrgan`, `basicsr`, and `torch`. If they are missing, the command exits with a clear error that points at the same install command.

This repo also shims the older `torchvision.transforms.functional_tensor` import that `basicsr` still expects, so recent `torchvision` builds are less likely to fail on that legacy path.

For live throughput, `bgremoval-upscale` now also sets camera capture buffering to one frame and supports `--virtualcam-no-sleep` to remove virtualcam pacing.

The live camera path uses a one-frame queue, so the capture thread keeps reading ahead while inference works on the latest available frame instead of building a backlog.

The official Real-ESRGAN repository documents `basicsr` as a required dependency and mentions `facexlib` and `gfpgan` for face enhancement. This repo does not enable face enhancement by default.
