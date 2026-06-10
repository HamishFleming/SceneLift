# Model Weights

This directory is reserved for local model weights and cached artifacts.

Suggested layout:

- `grabcut/` for classical config or tuning files
- `rembg/` for local rembg sessions or alternate weights
- `birefnet/` for locally mirrored BiRefNet checkpoints or exports
- `u2net-human-seg/` for cached `onnx/model.onnx` and `preprocessor_config.json` artifacts from U-2-Net-Human-Seg
- `mediapipe-selfie-segmentation/` for cached `onnx/model.onnx` and `preprocessor_config.json` artifacts from MediaPipe Selfie Segmentation

The built-in `birefnet` backend loads from Hugging Face by default, but the registry now has a stable place to attach local weights later.
The built-in `u2net-human-seg` backend downloads its ONNX model and processor config from Hugging Face by default and can be redirected to a local mirrored directory through the registry.
The built-in `mediapipe-selfie-segmentation` backend uses the same ONNXRuntime path and local cache structure.
