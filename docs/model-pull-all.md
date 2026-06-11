# Pull All Models

Use `bgremoval-model-pull-all` to download every model that currently has explicit fetch metadata in the registry, plus the Real-ESRGAN checkpoints that are published as direct release assets.

## What it pulls

- `u2net-human-seg`
- `mediapipe-selfie-segmentation`
- `modnet-trt`
- `ben2-trt`
- `realesrgan-x4plus`
- `realesrgan-x4plus-anime`

## What it skips

- `grabcut`
- `rembg`
- `birefnet`, which still loads on demand from Hugging Face and does not yet have a dedicated local mirror fetch path in this repo
- `realesrgan-x2plus`, which is available as a runtime option but does not yet have a mirrored release URL wired into the pull-all flow

## Example

```bash
bgremoval-model-pull-all
```

By default it writes into `src/bgremoval/models/weights/`. If you want to mirror into another directory first, pass:

```bash
bgremoval-model-pull-all --weights-root /path/to/weights
```

## Notes

- The command follows the registry metadata, so when you add a new fetchable model to `src/bgremoval/models/registry.py`, it becomes part of the pull-all workflow automatically.
- The upscaler registry is folded into the same command, so Real-ESRGAN weights with explicit release URLs are mirrored in the same pass.
