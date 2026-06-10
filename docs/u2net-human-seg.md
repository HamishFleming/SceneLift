# U-2-Net Human Seg

`u2net-human-seg` is a selectable background-removal backend backed by the Hugging Face model `BritishWerewolf/U-2-Net-Human-Seg`.

Use it like this:

```bash
bgremoval --input camera:0 --output virtualcam --method u2net-human-seg --live
```

Notes:

- The backend downloads `onnx/model.onnx` and `preprocessor_config.json` from Hugging Face and runs the ONNX graph directly.
- It uses ONNXRuntime, not `transformers`, because the repo ships a custom `u2net` architecture that stock Transformers does not resolve.
- The `hf` optional dependency group now includes `onnxruntime`.
- Cached assets live under `src/bgremoval/models/weights/u2net-human-seg/`.

Source:

- https://huggingface.co/BritishWerewolf/U-2-Net-Human-Seg
