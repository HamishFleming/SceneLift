# MediaPipe Selfie Segmentation

`mediapipe-selfie-segmentation` is a selectable ONNXRuntime background-removal backend backed by `onnx-community/mediapipe_selfie_segmentation`.

Use it like this:

```bash
bgremoval --input camera:0 --output virtualcam --method mediapipe-selfie-segmentation --live
```

Notes:

- The backend downloads `onnx/model.onnx` and `preprocessor_config.json` from Hugging Face.
- It runs the ONNX graph directly through ONNXRuntime, so it does not depend on `transformers` for model loading.
- The processor config uses a fixed `256x256` resize path with no normalization or padding.
- Cached assets live under `src/bgremoval/models/weights/mediapipe-selfie-segmentation/`.

Source:

- https://huggingface.co/onnx-community/mediapipe_selfie_segmentation
